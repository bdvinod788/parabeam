"""calibrate.py against a fake generator whose timers it must read back exactly."""
import asyncio
import json

import httpx
import pytest

from parabeam.dist.costmodel import CostModel
from parabeam.tools import calibrate

STEP_S = 0.004  # the fake decodes every token in this long
PREFILL_PER_TOKEN_S = 1e-5


class FakeServers:
    """Records each request's timers one /metrics read late, as vLLM may."""

    def __init__(self) -> None:
        self.decode = self.prefill = 0.0
        self.requests = 0
        self.pending: list[tuple[float, float]] = []
        self.prm_calls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            label = '{model_name="g"}'
            shown = (f"vllm:request_decode_time_seconds_sum{label} {self.decode}\n"
                     f"vllm:request_prefill_time_seconds_sum{label} {self.prefill}\n"
                     f"vllm:request_prefill_time_seconds_count{label} {self.requests}\n"
                     "vllm:num_preemptions_total 0\n")
            for decode, prefill in self.pending:
                self.decode += decode
                self.prefill += prefill
                self.requests += 1
            self.pending = []
            return httpx.Response(200, text=shown)
        body = json.loads(request.content)
        if request.url.path == "/pooling":
            self.prm_calls += 1
            return httpx.Response(200, json={"data": []})
        tokens = body["max_tokens"]
        self.pending.append(((tokens - 1) * STEP_S, len(body["prompt"]) * PREFILL_PER_TOKEN_S))
        return httpx.Response(200, json={"usage": {"completion_tokens": tokens}})


@pytest.fixture(scope="module")
def measured():
    fake = FakeServers()
    data = asyncio.run(calibrate.measure("http://gen", "http://prm",
                                         transport=httpx.MockTransport(fake.handler)))
    return fake, data


def test_decode_steps_come_from_the_servers_timers(measured):
    _, data = measured
    for row in data["decode_s"]:
        assert row == pytest.approx([STEP_S] * len(calibrate.BATCHES))
    assert data["suspect"] == []


def test_prefill_is_the_servers_prefill_time(measured):
    _, data = measured
    expected = [context * PREFILL_PER_TOKEN_S for context in calibrate.CONTEXTS]
    assert data["prefill_s"] == pytest.approx(expected)


def test_the_output_loads_as_a_cost_model(measured, tmp_path):
    fake, data = measured
    assert fake.prm_calls == calibrate.REPEATS
    path = tmp_path / "cost_model.json"
    path.write_text(json.dumps(data))
    assert CostModel.load(path).decode(64, 1000) == pytest.approx(STEP_S)


def test_impossible_steps_are_suspect():
    good = [[0.004 + 0.0001 * b + 0.001 * c for b in range(len(calibrate.BATCHES))]
            for c in range(len(calibrate.CONTEXTS))]
    assert calibrate.suspect(good) == []
    bad = [row[:] for row in good]
    bad[0][7] = 0.0004  # job 13956487: b = 128 at a tenth of b = 64
    bad[2][8] = 0.5 * bad[1][8]  # a longer context, twice as fast
    assert calibrate.suspect(bad) == [(500, 128), (2000, 256)]


def test_missing_timers_are_an_error():
    def handler(request):
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:other 1\n")
        return httpx.Response(200, json={"usage": {"completion_tokens": 1}})

    with pytest.raises(RuntimeError, match="does not export"):
        asyncio.run(calibrate.measure("http://gen", "http://prm",
                                      transport=httpx.MockTransport(handler)))
