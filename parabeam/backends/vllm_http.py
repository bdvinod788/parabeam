"""The Backend for real runs: one generator and one PRM vLLM server per rank.

Every request carries token IDs, never text, so prefixes match the servers'
prefix caches exactly. The PRM must be served through parabeam's LAST-pooling
plugin (see vllm_prm.py); it answers classify requests with the newest step's
reward and reads the prefix cache, which the stock PRM never does.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

from parabeam.backends.base import Aborted, Finish, GenReq, StepOut
from parabeam.backends.prm_ids import (
    POSITIVE,
    STEP_TAG,
    SYSTEM,
    Encode,
    probabilities,
    prompt_text,
    step_ids,
)
from parabeam.core.beam import Problem

MAX_MODEL_LEN = 4096  # both models' context
STOP = "\n\n"  # steps end at a blank line, as in search-and-learn
TEMPERATURE = 0.8
TOP_P = 1.0

# vLLM closes a connection idle for VLLM_HTTP_TIMEOUT_KEEP_ALIVE seconds (5 by
# default, the same as httpx's keep-alive), so a request sent on a connection the
# server is closing fails with a ReadError; job 13956488 died that way at N = 256.
# launch_servers.sh raises the server's limit, the client gives up its idle
# connections well before that, and a request whose connection drops is resent.
SERVER_KEEP_ALIVE_S = 300  # launch_servers.sh exports this to both servers
CLIENT_KEEP_ALIVE_S = 60
RETRIES = 3
DROPPED = (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError, httpx.ConnectError)


def http_client(timeout: float, transport: httpx.AsyncBaseTransport | None = None
                ) -> httpx.AsyncClient:
    """Return a client with no connection cap and a shorter keep-alive than the servers'.

    httpx allows 100 connections by default, which would cap a level at 100 beams.
    """
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None,
                          keepalive_expiry=CLIENT_KEEP_ALIVE_S)
    if transport is None:
        transport = httpx.AsyncHTTPTransport(retries=3, limits=limits)
    return httpx.AsyncClient(timeout=timeout, transport=transport, limits=limits)


async def post_json(client: httpx.AsyncClient, url: str, body: dict,
                    on_retry: Callable[[], None] | None = None) -> dict:
    """POST body and return the JSON reply, resending if the connection drops.

    Resending is safe: a request carries its own seed, and the reply to the
    dropped attempt was never seen.
    """
    for attempt in range(RETRIES + 1):
        try:
            reply = await client.post(url, json=body)
            break
        except DROPPED:
            if attempt == RETRIES:
                raise
            if on_retry is not None:
                on_retry()
            await asyncio.sleep(0.5 * 2**attempt)
    if reply.status_code != 200:
        raise RuntimeError(f"{url} returned {reply.status_code}: {reply.text[:500]}")
    return reply.json()


def finish_of(choice: dict) -> Finish:
    """Map a vLLM choice to stop, length or eos.

    vLLM reports finish_reason "stop" both for a stop string and for EOS; only
    a stop string sets stop_reason.
    """
    if choice["finish_reason"] == "length":
        return "length"
    if choice["finish_reason"] == "stop":
        return "stop" if choice.get("stop_reason") == STOP else "eos"
    raise RuntimeError(f"unexpected finish_reason {choice['finish_reason']!r}")


def step_budget(prefix_len: int, max_tokens: int, max_model_len: int = MAX_MODEL_LEN) -> int:
    """Return how many tokens a step may use without overflowing the context."""
    return max(0, min(max_tokens, max_model_len - prefix_len))


class VllmBackend:
    def __init__(
        self,
        gen_url: str,
        prm_url: str,
        prm_encode: Encode,
        tag_id: int,
        gen_prompt: Callable[[str], list[int]],
        *,
        max_model_len: int = MAX_MODEL_LEN,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 3600,
    ) -> None:
        self.gen_url, self.prm_url = gen_url.rstrip("/"), prm_url.rstrip("/")
        self._prm_encode = prm_encode
        self._tag_id = tag_id
        self._gen_prompt = gen_prompt
        self._max_model_len = max_model_len
        self._client = http_client(timeout, transport)
        self._running: dict[str, asyncio.Task] = {}
        self._aborted: set[str] = set()
        self.truncated = 0  # PRM inputs cut to fit the context
        self.retried = 0  # requests resent after their connection dropped

    @classmethod
    def from_models(cls, gen_url: str, prm_url: str, generator: Path, prm: Path) -> VllmBackend:
        """Load both tokenizers from the model directories the servers use."""
        from transformers import AutoTokenizer

        gen_tok = AutoTokenizer.from_pretrained(generator)
        prm_tok = AutoTokenizer.from_pretrained(prm)
        tag_id = prm_tok.convert_tokens_to_ids(STEP_TAG)
        if tag_id is None or tag_id == prm_tok.unk_token_id:
            raise ValueError(f"{STEP_TAG} is not in the PRM's vocabulary")

        def gen_prompt(question: str) -> list[int]:
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": question}]
            text = gen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return gen_tok.encode(text, add_special_tokens=False)

        return cls(gen_url, prm_url, lambda text: prm_tok.encode(text, add_special_tokens=False),
                   tag_id, gen_prompt)

    def problem(self, problem_id: int, question: str) -> Problem:
        return Problem(
            problem_id,
            tuple(self._gen_prompt(question)),
            tuple(self._prm_encode(prompt_text(question))),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, url: str, body: dict) -> dict:
        return await post_json(self._client, url, body, self._count_retry)

    def _count_retry(self) -> None:
        self.retried += 1

    async def _complete(self, req: GenReq) -> StepOut:
        t_start = time.perf_counter()
        budget = step_budget(len(req.prefix_ids), req.max_tokens, self._max_model_len)
        if budget == 0:  # the context is full, so the beam ends as if cut off
            return StepOut([], "", "length", t_start, t_start)
        body = {
            "model": "generator",
            "prompt": list(req.prefix_ids),
            "max_tokens": budget,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": req.seed,
            "return_token_ids": True,
            "skip_special_tokens": True,
        }
        if not req.final:  # the last level generates to EOS
            body["stop"] = [STOP]
            body["include_stop_str_in_output"] = True  # beams keep the blank line's tokens
        choice = (await self._post(f"{self.gen_url}/v1/completions", body))["choices"][0]
        if choice.get("token_ids") is None:
            raise RuntimeError("the generator did not return token IDs")
        return StepOut(list(choice["token_ids"]), choice["text"], finish_of(choice),
                       t_start, time.perf_counter())

    async def generate(self, req: GenReq) -> StepOut:
        if req.beam_id in self._running:
            raise ValueError(f"{req.beam_id} is already running")
        task = asyncio.ensure_future(self._complete(req))
        self._running[req.beam_id] = task
        try:
            return await task
        except asyncio.CancelledError:
            if req.beam_id in self._aborted:  # cancelled by abort, not by our caller
                raise Aborted(req.beam_id) from None
            raise
        finally:
            self._aborted.discard(req.beam_id)
            if self._running.get(req.beam_id) is task:
                del self._running[req.beam_id]

    def prm_step_ids(self, out: StepOut) -> list[int]:
        return step_ids(self._prm_encode, out.text, self._tag_id)

    async def score(self, prm_ids: list[list[int]]) -> list[float]:
        if not prm_ids:
            return []
        # A full-length beam can tokenize a few tokens longer for the PRM than for
        # the generator. Rather than fail the run, keep the newest tokens, which
        # hold the step being scored, and count it.
        inputs = []
        for ids in prm_ids:
            if len(ids) > self._max_model_len:
                self.truncated += 1
                ids = ids[-self._max_model_len :]
            inputs.append(list(ids))
        body = {"model": "prm", "input": inputs, "task": "classify"}
        data = sorted((await self._post(f"{self.prm_url}/pooling", body))["data"],
                      key=lambda item: item["index"])
        if len(data) != len(prm_ids):
            raise RuntimeError(f"asked for {len(prm_ids)} scores, got {len(data)}")
        return [probabilities(_last_row(item["data"]))[POSITIVE] for item in data]

    def abort(self, beam_id: str) -> None:
        task = self._running.get(beam_id)
        if task is not None and not task.done():
            self._aborted.add(beam_id)
            task.cancel()  # closing the connection makes vLLM abort the request


def _last_row(data: Sequence) -> Sequence[float]:
    """classify returns one row; accept a list of rows too, taking the last."""
    return data[-1] if data and isinstance(data[0], list) else data
