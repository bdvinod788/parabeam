"""The vLLM client, against an in-process fake of both servers."""
import asyncio
import json
import math

import httpx
import pytest

from parabeam.backends.base import Aborted, GenReq
from parabeam.backends.prm_ids import prompt_text
from parabeam.backends.vllm_http import VllmBackend, finish_of, step_budget

TAG = 99
encode = lambda text: [ord(c) % 90 for c in text]  # noqa: E731


def backend(handler, **kwargs):
    return VllmBackend("http://gen", "http://prm", encode, TAG, lambda q: [7, *encode(q)],
                       transport=httpx.MockTransport(handler), **kwargs)


def req(final=False, prefix=(1, 2, 3), max_tokens=2048, beam_id="p1/L1/0"):
    return GenReq(beam_id, list(prefix), 12345, max_tokens, final)


def completion(token_ids, text, finish_reason="stop", stop_reason="\n\n"):
    return httpx.Response(200, json={"choices": [{
        "text": text, "token_ids": token_ids, "finish_reason": finish_reason,
        "stop_reason": stop_reason}]})


@pytest.mark.parametrize("choice, finish", [
    ({"finish_reason": "stop", "stop_reason": "\n\n"}, "stop"),
    ({"finish_reason": "stop", "stop_reason": None}, "eos"),
    ({"finish_reason": "stop"}, "eos"),
    ({"finish_reason": "length", "stop_reason": None}, "length"),
])
def test_finish_reasons(choice, finish):
    assert finish_of(choice) == finish


def test_an_unknown_finish_reason_is_an_error():
    with pytest.raises(RuntimeError):
        finish_of({"finish_reason": "abort"})


def test_the_step_budget_never_overflows_the_context():
    assert step_budget(100, 2048) == 2048
    assert step_budget(3000, 2048) == 1096
    assert step_budget(4096, 2048) == 0
    assert step_budget(5000, 2048) == 0


def test_a_step_request_sends_token_ids_and_stops_at_a_blank_line():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        assert request.url == "http://gen/v1/completions"
        return completion([5, 6, 271], "x = 1\n\n")

    out = asyncio.run(backend(handler).generate(req()))
    assert seen["prompt"] == [1, 2, 3] and seen["seed"] == 12345
    assert seen["stop"] == ["\n\n"] and seen["include_stop_str_in_output"] is True
    assert seen["return_token_ids"] is True and seen["max_tokens"] == 2048
    assert seen["temperature"] == 0.8 and seen["top_p"] == 1.0
    assert (out.token_ids, out.text, out.finish) == ([5, 6, 271], "x = 1\n\n", "stop")
    assert out.t_end >= out.t_start


def test_the_last_level_generates_to_eos():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return completion([5], "done", stop_reason=None)

    out = asyncio.run(backend(handler).generate(req(final=True)))
    assert "stop" not in seen and out.finish == "eos"


def test_a_long_prefix_gets_a_smaller_cap():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return completion([5], "x", "length", None)

    asyncio.run(backend(handler).generate(req(prefix=[1] * 4000)))
    assert seen["max_tokens"] == 96


def test_a_full_context_ends_the_beam_without_a_request():
    def handler(request):
        raise AssertionError("no request expected")

    out = asyncio.run(backend(handler).generate(req(prefix=[1] * 4096)))
    assert (out.token_ids, out.finish) == ([], "length")


def test_missing_token_ids_are_an_error():
    def handler(request):
        return completion(None, "x")

    with pytest.raises(RuntimeError):
        asyncio.run(backend(handler).generate(req()))


def test_a_server_error_is_reported():
    def handler(request):
        return httpx.Response(400, text="bad request")

    with pytest.raises(RuntimeError, match="400"):
        asyncio.run(backend(handler).generate(req()))


def test_scores_come_back_in_input_order_as_probabilities():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        assert request.url == "http://prm/pooling"
        return httpx.Response(200, json={"data": [
            {"index": 1, "data": [0.0, math.log(3)]},  # logits
            {"index": 0, "data": [0.2, 0.8]},
        ]})

    scores = asyncio.run(backend(handler).score([[1, TAG], [2, TAG]]))
    assert seen["task"] == "classify" and seen["model"] == "prm"
    assert scores == pytest.approx([0.8, 0.75])


def test_a_short_score_reply_is_an_error():
    def handler(request):
        return httpx.Response(200, json={"data": [{"index": 0, "data": [0.5, 0.5]}]})

    with pytest.raises(RuntimeError):
        asyncio.run(backend(handler).score([[1], [2]]))


def test_an_oversize_prm_input_keeps_its_newest_tokens():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "data": [0.5, 0.5]}]})

    client = backend(handler, max_model_len=8)
    asyncio.run(client.score([list(range(10))]))
    assert seen["input"] == [list(range(2, 10))] and client.truncated == 1


def test_no_inputs_need_no_request():
    def handler(request):
        raise AssertionError("no request expected")

    assert asyncio.run(backend(handler).score([])) == []


def test_prm_step_ids_drop_the_blank_line_and_add_the_tag():
    from parabeam.backends.base import StepOut

    ids = backend(lambda r: None).prm_step_ids(StepOut([1], "ab\n\n", "stop", 0, 0))
    assert ids == [*encode("ab"), TAG]


def test_problems_get_both_prompts():
    problem = backend(lambda r: None).problem(4, "Why?")
    assert problem.id == 4
    assert problem.gen_prompt_ids == (7, *encode("Why?"))
    assert problem.prm_prompt_ids == tuple(encode(prompt_text("Why?")))


def slow_handler():
    async def handler(request):
        await asyncio.sleep(30)
        return completion([1], "x")

    return handler


def test_abort_cancels_a_running_request():
    async def run():
        client = backend(slow_handler())
        task = asyncio.create_task(client.generate(req()))
        await asyncio.sleep(0.01)
        client.abort("p1/L1/0")
        with pytest.raises(Aborted):
            await asyncio.wait_for(task, 5)
        client.abort("p1/L1/0")  # finished: does nothing
        return client._running

    assert asyncio.run(run()) == {}


def test_cancelling_the_caller_is_not_an_abort():
    async def run():
        client = backend(slow_handler())
        task = asyncio.create_task(client.generate(req()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return client._running

    assert asyncio.run(run()) == {}


def test_a_running_duplicate_is_refused():
    async def run():
        client = backend(slow_handler())
        task = asyncio.create_task(client.generate(req()))
        await asyncio.sleep(0.01)
        with pytest.raises(ValueError):
            await client.generate(req())
        task.cancel()

    asyncio.run(run())


def flaky(failures, error=httpx.ReadError):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) <= failures:
            raise error("connection dropped", request=request)
        return completion([5, 271], "a\n\n")

    return handler, calls


@pytest.fixture
def no_backoff(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr("parabeam.backends.vllm_http.asyncio.sleep",
                        lambda seconds: real_sleep(0))


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.RemoteProtocolError])
def test_a_dropped_connection_is_resent_and_counted(no_backoff, error):
    handler, calls = flaky(2, error)
    b = backend(handler)
    out = asyncio.run(b.generate(req()))
    assert out.token_ids == [5, 271] and len(calls) == 3 and b.retried == 2
    assert json.loads(calls[0].content) == json.loads(calls[2].content)  # same seed


def test_a_connection_that_keeps_dropping_fails(no_backoff):
    from parabeam.backends.vllm_http import RETRIES

    handler, calls = flaky(RETRIES + 1)
    with pytest.raises(httpx.ReadError):
        asyncio.run(backend(handler).generate(req()))
    assert len(calls) == RETRIES + 1


def test_a_timeout_is_not_resent(no_backoff):
    handler, calls = flaky(1, httpx.ReadTimeout)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(backend(handler).generate(req()))
    assert len(calls) == 1


def test_the_client_lets_idle_connections_go_before_the_servers_do():
    import re
    from pathlib import Path

    from parabeam.backends import vllm_http

    launcher = (Path(__file__).parents[1] / "slurm/launch_servers.sh").read_text()
    server = int(re.search(r"VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-(\d+)", launcher).group(1))
    assert server == vllm_http.SERVER_KEEP_ALIVE_S
    assert vllm_http.CLIENT_KEEP_ALIVE_S < server
