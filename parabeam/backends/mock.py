"""A CPU stand-in for the two vLLM servers, for tests, the demo and grading.

What a request produces depends only on its seed and path, and a score only on
its token IDs, so results cannot vary with rank, batch or timing. If two policies
disagree on the mock, the fault is in the policy. Step lengths and solution
lengths follow PRM800K (see tools/fit_steps.py); time comes from a CostModel.

Time passes in decode ticks, as in continuous batching: each tick extends every
active request by one token and takes t(batch), plus the prefill of any request
that joined. With scale=0 the clock is simulated and runs as fast as the CPU can.
With scale>0 the mock really sleeps, for simulated time times scale.

Chaos changes when things happen, never what they produce. It counts delays in
event-loop turns, not seconds, so a failure replays from its seed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from array import array
from dataclasses import dataclass
from pathlib import Path

from parabeam.backends.base import Aborted, Finish, GenReq, StepOut
from parabeam.core.beam import Problem
from parabeam.dist.costmodel import GUESS, CostModel

_STATS = json.loads(Path(__file__).with_name("step_stats.json").read_text())
STEP_TOKENS: list[int] = _STATS["step_tokens"]
PROBLEM_STEPS: list[int] = _STATS["problem_steps"]
STOP_SPREAD: int = _STATS["stop_spread"]

EXTRA_0 = 1 << 20  # above every mock token, which fit in 16 bits
ANSWERS = 4  # distinct answers, so that votes are contested
FINAL_STEPS = 5  # the last level generates to EOS, a few steps at once
CACHE_BLOCK = 16  # tokens per block, as in vLLM
CACHE_BLOCKS = 1 << 18  # a bound for long runs; calibrate against the real KV cache


def _draw(seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _step_length(seed: int, salt: str) -> int:
    return STEP_TOKENS[int(_draw(seed, salt) * len(STEP_TOKENS))]


def stop_chance(problem_id: int, level: int) -> float:
    """Return P(a step at this level ends its solution | the solution reached it).

    Solutions to a problem end uniformly within STOP_SPREAD levels of its typical
    length. Stopping cannot be independent of the problem: with N children per
    level, some would always continue and no search would end before max_depth.
    """
    typical = PROBLEM_STEPS[int(_draw(problem_id, "steps") * len(PROBLEM_STEPS))]
    first, last = max(1, typical - STOP_SPREAD), typical + STOP_SPREAD
    return 0.0 if level < first else 1 / max(1, last - level + 1)


def mock_step(req: GenReq) -> tuple[list[int], str, Finish]:
    """Return the tokens, text and finish reason of a request."""
    seed = req.seed
    problem, level_key, _ = req.beam_id.split("/")
    if req.final:
        length = sum(_step_length(seed, f"length{i}") for i in range(FINAL_STEPS))
        stop = True
    else:
        length = _step_length(seed, "length")
        chance = stop_chance(int(problem.removeprefix("p")), int(level_key.removeprefix("L")))
        stop = _draw(seed, "stop") < chance

    if length > req.max_tokens:
        length, finish, text = req.max_tokens, "length", f"Step {seed:x}"
    elif stop:
        answer = int(_draw(seed, "answer") * ANSWERS)
        finish, text = "eos", f"Step {seed:x}. The answer is \\boxed{{{answer}}}."
    else:
        finish, text = "stop", f"Step {seed:x}.\n\n"
    rng = random.Random(seed)
    return [rng.getrandbits(16) for _ in range(length)], text, finish


def mock_score(prm_ids: list[int]) -> float:
    """Return a reward in [0, 1) fixed by the IDs, which identify the path."""
    digest = hashlib.sha256(array("Q", prm_ids).tobytes()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def mock_problem(problem_id: int) -> Problem:
    rng = random.Random(problem_id)
    size = rng.randrange(40, 240)
    return Problem(
        problem_id,
        tuple(rng.getrandbits(16) for _ in range(size)),
        tuple(rng.getrandbits(16) for _ in range(size + 8)),
    )


class PrefixCache:
    """Hashed blocks of tokens, each chained to the last, as in vLLM."""

    def __init__(self, capacity: int = CACHE_BLOCKS) -> None:
        self._blocks: dict[int, None] = {}  # oldest first
        self._capacity = capacity

    @staticmethod
    def _chain(ids: list[int]) -> list[int]:
        chain, block = [], 0
        for i in range(0, len(ids) - CACHE_BLOCK + 1, CACHE_BLOCK):
            block = hash((block, *ids[i : i + CACHE_BLOCK]))
            chain.append(block)
        return chain

    def cached(self, ids: list[int]) -> int:
        """Return how many leading tokens of ids are cached."""
        count = 0
        for block in self._chain(ids):
            if block not in self._blocks:
                break
            count += CACHE_BLOCK
        return count

    def add(self, ids: list[int]) -> None:
        for block in self._chain(ids):
            self._blocks.pop(block, None)
            self._blocks[block] = None
        while len(self._blocks) > self._capacity:
            del self._blocks[next(iter(self._blocks))]


@dataclass(frozen=True, slots=True)
class Chaos:
    """Random holds and uneven ticks, to shake out order and timing bugs."""

    seed: int
    max_hold: int = 64  # loop turns a request can wait on its way in or out
    stall_chance: float = 0.1  # share of holds that are 16 times longer
    jitter: float = 1.0  # a tick can take up to 1 + jitter times as long


@dataclass(slots=True, eq=False)
class _Job:
    beam_id: str
    remaining: int
    context: int
    done: asyncio.Future[None]
    joined: bool = False


class MockBackend:
    """One rank's pair of servers."""

    def __init__(
        self,
        cost: CostModel = GUESS,
        chaos: Chaos | None = None,
        scale: float = 0.0,
    ) -> None:
        self._cost = cost
        self._chaos = chaos
        self._scale = scale
        self._jitter = random.Random(chaos.seed if chaos else 0)
        self._gen_cache = PrefixCache()
        self._prm_cache = PrefixCache()
        self._jobs: dict[str, _Job] = {}
        self._ticker: asyncio.Task[None] | None = None
        self._prefill_s = 0.0
        self._prm_lock = asyncio.Lock()
        self._scores = 0
        self._simulated = 0.0
        self._started = time.perf_counter()

    def now(self) -> float:
        """Return simulated seconds since this backend was created."""
        if self._scale:
            return (time.perf_counter() - self._started) / self._scale
        return self._simulated

    def advance_to(self, t: float) -> None:
        """Idle until simulated time t, as a rank does at a barrier.

        Ranks in a run have one backend each, so their simulated clocks drift
        apart; schedulers call this with the slowest rank's time when they meet.
        Real time needs no help, so with scale>0 this does nothing.
        """
        if not self._scale:
            self._simulated = max(self._simulated, t)

    async def _elapse(self, seconds: float) -> None:
        if self._scale:
            await asyncio.sleep(seconds * self._scale)
        else:
            self._simulated += seconds
            await asyncio.sleep(0)

    async def _hold(self, key: str) -> None:
        if self._chaos is None:
            return
        rng = random.Random(f"{self._chaos.seed}/{key}")
        turns = rng.randrange(self._chaos.max_hold)
        if rng.random() < self._chaos.stall_chance:
            turns *= 16
        for _ in range(turns):
            await asyncio.sleep(0)

    def _join(self, job: _Job, prefix: list[int]) -> None:
        self._prefill_s += self._cost.prefill(len(prefix) - self._gen_cache.cached(prefix))
        self._gen_cache.add(prefix)
        job.joined = True
        if self._ticker is None or self._ticker.done():
            self._ticker = asyncio.create_task(self._tick())

    async def _tick(self) -> None:
        while batch := [job for job in self._jobs.values() if job.joined]:
            context = sum(job.context for job in batch) / len(batch)
            seconds = self._cost.decode(len(batch), context) + self._prefill_s
            self._prefill_s = 0.0
            if self._chaos:
                seconds *= 1 + self._jitter.random() * self._chaos.jitter
            await self._elapse(seconds)
            for job in batch:
                if job.done.done():
                    continue  # aborted or cancelled during the tick
                job.remaining -= 1
                job.context += 1
                if job.remaining == 0:
                    del self._jobs[job.beam_id]
                    job.done.set_result(None)

    async def generate(self, req: GenReq) -> StepOut:
        if req.beam_id in self._jobs:
            raise ValueError(f"{req.beam_id} is already running")
        t_start = self.now()
        token_ids, text, finish = mock_step(req)
        done = asyncio.get_running_loop().create_future()
        job = _Job(req.beam_id, len(token_ids), len(req.prefix_ids), done)
        self._jobs[req.beam_id] = job
        try:
            await self._hold(f"{req.beam_id}/in")
            if not done.done():
                self._join(job, req.prefix_ids)
            await done
        finally:
            if self._jobs.get(req.beam_id) is job:
                del self._jobs[req.beam_id]
        self._gen_cache.add(req.prefix_ids + token_ids)
        t_end = self.now()
        await self._hold(f"{req.beam_id}/out")
        return StepOut(token_ids, text, finish, t_start, t_end)

    def prm_step_ids(self, out: StepOut) -> list[int]:
        return [*out.token_ids, EXTRA_0]

    async def score(self, prm_ids: list[list[int]]) -> list[float]:
        self._scores += 1
        await self._hold(f"score/{self._scores}")
        tokens = 0
        for ids in prm_ids:
            tokens += len(ids) - self._prm_cache.cached(ids)
            self._prm_cache.add(ids)
        async with self._prm_lock:
            await self._elapse(tokens * self._cost.prm_s_per_token)
        return [mock_score(ids) for ids in prm_ids]

    def abort(self, beam_id: str) -> None:
        job = self._jobs.pop(beam_id, None)
        if job is not None:
            job.done.set_exception(Aborted(beam_id))
