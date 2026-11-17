"""frontend_bench against a fake generator that reports its own timers."""
import asyncio
import json

import httpx

from parabeam.tools import frontend_bench as fb


class FakeGenerator:
    def __init__(self) -> None:
        self.e2e = self.queue = 0.0
        self.count = 0
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            lines = [f'{name}_sum{{model_name="g"}} {self.e2e if key == "e2e" else self.queue}'
                     for key, name in fb.HISTOGRAMS.items()]
            lines.append(f'{fb.HISTOGRAMS["e2e"]}_count{{model_name="g"}} {self.count}')
            return httpx.Response(200, text="\n".join(lines) + "\n")
        body = json.loads(request.content)
        self.bodies.append(body)
        self.e2e += 0.5
        self.queue += 0.25
        self.count += 1
        n = body.get("n", 1)
        return httpx.Response(200, json={"usage": {"completion_tokens": n * fb.TOKENS},
                                         "choices": [{"text": "x", "token_ids": [1]}] * n})


def test_each_variant_sends_a_level_and_reads_the_servers_timers(tmp_path):
    fake = FakeGenerator()
    rows = asyncio.run(fb.bench("http://gen", "test", tmp_path / "b.jsonl", None, lengths=(8,),
                                repeats=1, transport=httpx.MockTransport(fake.handler)))
    assert [row["variant"] for row in rows] == ["real", "minimal", "grouped"]
    by = {row["variant"]: row for row in rows}
    assert by["real"]["requests"] == fb.BATCH and by["grouped"]["requests"] == fb.BATCH // fb.GROUP
    assert all(row["tokens"] == fb.BATCH * fb.TOKENS for row in rows)
    for row in rows:
        assert row["server_requests"] == row["requests"]
        assert row["server_mean_s"]["e2e"] == 0.5 and row["server_mean_s"]["queue"] == 0.25
    assert len((tmp_path / "b.jsonl").read_text().splitlines()) == 3


def test_bodies_match_what_the_search_sends():
    groups = fb.prefixes(16, __import__("random").Random(0))
    real = fb.requests_for(groups, "real")
    assert len(real) == fb.BATCH and real[0]["prompt"] == real[fb.GROUP - 1]["prompt"]
    assert real[0]["prompt"] != real[fb.GROUP]["prompt"]
    assert len({r["seed"] for r in real[: fb.GROUP]}) == fb.GROUP
    assert real[0]["stop"] == ["\n\n"] and real[0]["return_token_ids"]
    minimal = fb.requests_for(groups, "minimal")[0]
    assert "stop" not in minimal and minimal["min_tokens"] == fb.TOKENS
    grouped = fb.requests_for(groups, "grouped")
    assert len(grouped) == fb.BATCH // fb.GROUP and grouped[0]["n"] == fb.GROUP
