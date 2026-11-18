"""The aiohttp transport the vLLM client sends through, against a real local socket."""
import asyncio
import json
import time

import httpx
import pytest

from parabeam.backends.base import Aborted, GenReq
from parabeam.backends.vllm_http import AiohttpTransport, VllmBackend, http_client

pytest.importorskip("aiohttp")


def reply_for(body: dict) -> bytes:
    return json.dumps({"choices": [{"text": "a\n\n", "token_ids": [5, body["seed"]],
                                    "finish_reason": "stop", "stop_reason": "\n\n"}]}).encode()


class Server:
    """HTTP/1.1 keep-alive server; `behave(n, body)` picks what the n-th request gets."""

    def __init__(self, behave=lambda n, body: "ok", delay=0.0):
        self.behave, self.delay = behave, delay
        self.requests, self.connections, self.closed = [], 0, 0

    async def handle(self, reader, writer):
        self.connections += 1
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                size = next(int(line.split(b":")[1]) for line in head.split(b"\r\n")
                            if line.lower().startswith(b"content-length:"))
                body = json.loads(await reader.readexactly(size))
                self.requests.append(body)
                action = self.behave(len(self.requests), body)
                if action == "drop":
                    break
                if action == "hang":
                    await reader.read()  # until the client closes the connection
                    break
                await asyncio.sleep(self.delay)
                out = reply_for(body)
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                             b"content-length: %d\r\n\r\n%s" % (len(out), out))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except asyncio.CancelledError:  # the server is shutting down
            return
        self.closed += 1
        writer.close()

    async def __aenter__(self):
        self._server = await asyncio.start_server(self.handle, "127.0.0.1", 0, backlog=1024)
        self.url = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        self._server.close()


def backend(url, **kwargs):
    return VllmBackend(url, url, lambda text: [1], 99, lambda q: [7], **kwargs)


def req(seed=12345, beam_id="p1/L1/0"):
    return GenReq(beam_id, [1, 2, 3], seed, 64, False)


@pytest.fixture
def no_backoff(monkeypatch):  # patches asyncio.sleep everywhere, the server's too
    real_sleep = asyncio.sleep
    monkeypatch.setattr("parabeam.backends.vllm_http.asyncio.sleep",
                        lambda seconds: real_sleep(0))


def test_the_default_client_sends_through_aiohttp():
    assert isinstance(http_client(10)._transport, AiohttpTransport)


def test_a_step_round_trips_with_its_body_intact():
    async def run():
        async with Server() as server:
            b = backend(server.url)
            out = await b.generate(req())
            await b.aclose()
        assert out.token_ids == [5, 12345] and out.finish == "stop" and out.text == "a\n\n"
        body = server.requests[0]
        assert body["prompt"] == [1, 2, 3] and body["stop"] == ["\n\n"]

    asyncio.run(run())


def test_connections_are_reused_across_levels():
    async def run():
        async with Server(delay=0.01) as server:
            b = backend(server.url)
            for level in range(3):
                await asyncio.gather(*(b.generate(req(i, f"b{level}/{i}")) for i in range(32)))
            await b.aclose()
        assert len(server.requests) == 96 and server.connections == 32

    asyncio.run(run())


def test_a_dropped_connection_is_resent(no_backoff):
    async def run():
        async with Server(lambda n, body: "drop" if n == 1 else "ok") as server:
            b = backend(server.url)
            out = await b.generate(req())
            await b.aclose()
        assert out.token_ids == [5, 12345] and b.retried == 1 and len(server.requests) == 2

    asyncio.run(run())


def test_a_refused_connection_is_a_connect_error(no_backoff):
    async def run():
        async with Server() as server:
            url = server.url
        b = backend(url)  # the server has gone
        with pytest.raises(httpx.ConnectError):
            await b.generate(req())
        await b.aclose()

    asyncio.run(run())


def test_a_slow_reply_is_a_read_timeout_and_not_resent():
    async def run():
        async with Server(delay=2) as server:
            b = backend(server.url, timeout=0.2)
            with pytest.raises(httpx.ReadTimeout):
                await b.generate(req())
            await b.aclose()
        assert len(server.requests) == 1

    asyncio.run(run())


def test_an_abort_closes_the_connection_so_vllm_aborts_the_request():
    async def run():
        async with Server(lambda n, body: "hang") as server:
            b = backend(server.url)
            task = asyncio.ensure_future(b.generate(req()))
            while not server.requests:
                await asyncio.sleep(0.01)
            b.abort("p1/L1/0")
            with pytest.raises(Aborted):
                await task
            for _ in range(100):
                if server.closed:
                    break
                await asyncio.sleep(0.01)
            await b.aclose()
        assert server.closed == 1

    asyncio.run(run())


def test_a_level_of_256_requests_costs_little_client_time():
    """httpx's own pool took 11 s for the second such batch; aiohttp takes about 0.1 s."""
    async def run():
        async with Server(delay=0.05) as server:
            b = backend(server.url)
            walls = []
            for level in range(3):
                start = time.perf_counter()
                await asyncio.gather(*(b.generate(req(i, f"b{level}/{i}")) for i in range(256)))
                walls.append(time.perf_counter() - start)
            await b.aclose()
        assert max(walls[1:]) < 2.0, walls

    asyncio.run(run())
