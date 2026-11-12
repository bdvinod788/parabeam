"""Ranks and the messages between them, for asyncio code.

Schedulers talk only to a Comm: one collective, allgather, and tagged
point-to-point messages. MpiComm runs over mpi4py without ever blocking the
event loop. It starts non-blocking MPI calls and polls them about every
millisecond, and a background task probes for incoming messages just as often,
so requests to the vLLM servers keep flowing while a rank waits. LocalWorld runs
P ranks as tasks in one event loop, for tests and the mock; with a chaos seed it
delays every delivery by a random number of loop turns.

Both pickle every message, so ranks never share an object, and both count the
bytes they send.

Collectives must be called by every rank in the same order. A rank calling a
second one before the first returns is a bug, so it raises rather than queueing.
"""
from __future__ import annotations

import asyncio
import pickle
import random
import sys
import traceback
from array import array
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from itertools import accumulate
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

POLL_S = 0.001  # how often a waiting rank polls MPI
MOVE = 1  # tag: beams sent to the rank that expands them


class Comm(Protocol):
    rank: int
    size: int
    sent_bytes: int  # pickled bytes this rank has sent, collectives included

    async def allgather(self, obj: T) -> list[T]:
        """Return every rank's obj, in rank order."""
        ...

    async def send(self, obj: Any, dest: int, tag: int) -> None: ...

    async def recv(self, tag: int) -> tuple[int, Any]:
        """Return (source, obj) for the next message with this tag, from any rank."""
        ...


def _dumps(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


class _OneCollective:
    def __init__(self) -> None:
        self._busy = False

    @contextmanager
    def __call__(self) -> Iterator[None]:
        if self._busy:
            raise RuntimeError("a collective is already running; ranks must call them in order")
        self._busy = True
        try:
            yield
        finally:
            self._busy = False


class MpiComm:
    """A Comm over mpi4py. Create it once per process and close it at the end."""

    def __init__(self, comm: Any = None, poll: float = POLL_S) -> None:
        from mpi4py import MPI

        self._MPI = MPI
        # A private context, so no other library's messages can match our probes.
        self._comm = (MPI.COMM_WORLD if comm is None else comm).Dup()
        self.rank, self.size = self._comm.Get_rank(), self._comm.Get_size()
        self.sent_bytes = 0
        self._poll = poll
        self._inbox: defaultdict[int, asyncio.Queue[tuple[int, Any]]] = defaultdict(asyncio.Queue)
        self._pump: asyncio.Task[None] | None = None
        self._collective = _OneCollective()

    async def _wait(self, request: Any) -> None:
        while not request.Test():
            await asyncio.sleep(self._poll)

    def _start_pump(self) -> asyncio.Task[None]:
        if self._pump is None:
            self._pump = asyncio.get_running_loop().create_task(self._receive_forever())
        return self._pump

    async def _receive_forever(self) -> None:
        MPI = self._MPI
        status = MPI.Status()
        while True:
            message = self._comm.Improbe(MPI.ANY_SOURCE, MPI.ANY_TAG, status)
            if message is None:
                await asyncio.sleep(self._poll)
                continue
            source, tag = status.Get_source(), status.Get_tag()
            buf = bytearray(status.Get_count(MPI.BYTE))
            await self._wait(message.Irecv([buf, MPI.BYTE]))
            self._inbox[tag].put_nowait((source, pickle.loads(buf)))

    async def allgather(self, obj: T) -> list[T]:
        MPI = self._MPI
        self._start_pump()
        with self._collective():
            data = _dumps(obj)
            sizes = array("q", [0]) * self.size
            await self._wait(self._comm.Iallgather([array("q", [len(data)]), MPI.INT64_T],
                                                   [sizes, MPI.INT64_T]))
            counts = list(sizes)
            offsets = list(accumulate(counts, initial=0))[:-1]
            out = bytearray(sum(counts))
            await self._wait(self._comm.Iallgatherv([data, MPI.BYTE],
                                                    [out, (counts, offsets), MPI.BYTE]))
            self.sent_bytes += len(data) * (self.size - 1)
            view = memoryview(out)
            return [pickle.loads(view[o : o + n]) for o, n in zip(offsets, counts, strict=True)]

    async def send(self, obj: Any, dest: int, tag: int) -> None:
        self._start_pump()
        data = _dumps(obj)
        self.sent_bytes += len(data)
        await self._wait(self._comm.Isend([data, self._MPI.BYTE], dest, tag))

    async def recv(self, tag: int) -> tuple[int, Any]:
        pump = self._start_pump()
        get = asyncio.ensure_future(self._inbox[tag].get())
        await asyncio.wait({get, pump}, return_when=asyncio.FIRST_COMPLETED)
        if not get.done():  # the pump died; raise its error rather than hang
            get.cancel()
            pump.result()
            raise RuntimeError("the MPI receive loop stopped")
        return get.result()

    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            try:
                await self._pump
            except asyncio.CancelledError:
                pass
            self._pump = None
        self._comm.Free()


@contextmanager
def abort_on_error() -> Iterator[None]:
    """Kill every rank if this one fails, instead of leaving them blocked in a collective.

    Without this, one rank's exception leaves the others waiting until Slurm's
    time limit.
    """
    try:
        yield
    except BaseException as error:
        if isinstance(error, SystemExit) and not error.code:
            raise
        from mpi4py import MPI

        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)


class LocalWorld:
    """P ranks in one event loop. Create it inside the loop that runs them."""

    def __init__(self, size: int, chaos: int | None = None, max_hold: int = 64) -> None:
        if size < 1:
            raise ValueError("a world needs at least one rank")
        self.size = size
        self.comms = [LocalComm(self, rank) for rank in range(size)]
        self._chaos = chaos
        self._max_hold = max_hold
        self._rounds: dict[int, tuple[list[bytes | None], asyncio.Future[None]]] = {}
        self._inbox = [defaultdict(asyncio.Queue) for _ in range(size)]
        self._deliveries: set[asyncio.Task[None]] = set()
        self._sent = 0

    async def _hold(self, key: str) -> None:
        if self._chaos is None:
            return
        for _ in range(random.Random(f"{self._chaos}/{key}").randrange(self._max_hold)):
            await asyncio.sleep(0)

    async def _allgather(self, rank: int, round_: int, data: bytes) -> list[bytes]:
        if round_ not in self._rounds:
            self._rounds[round_] = ([None] * self.size, asyncio.get_running_loop().create_future())
        values, everyone = self._rounds[round_]
        values[rank] = data
        if all(value is not None for value in values):
            del self._rounds[round_]
            everyone.set_result(None)
        await everyone
        await self._hold(f"gather/{round_}/{rank}")
        return values  # type: ignore[return-value]

    def _send(self, source: int, dest: int, tag: int, data: bytes) -> None:
        if not 0 <= dest < self.size:
            raise ValueError(f"no rank {dest} in a world of {self.size}")
        self._sent += 1
        key = f"send/{self._sent}/{source}/{dest}/{tag}"

        async def deliver() -> None:
            await self._hold(key)
            self._inbox[dest][tag].put_nowait((source, data))

        task = asyncio.get_running_loop().create_task(deliver())
        self._deliveries.add(task)
        task.add_done_callback(self._deliveries.discard)


class LocalComm:
    def __init__(self, world: LocalWorld, rank: int) -> None:
        self._world = world
        self.rank, self.size = rank, world.size
        self.sent_bytes = 0
        self._rounds = 0
        self._collective = _OneCollective()

    async def allgather(self, obj: T) -> list[T]:
        with self._collective():
            data = _dumps(obj)
            self.sent_bytes += len(data) * (self.size - 1)
            self._rounds += 1
            values = await self._world._allgather(self.rank, self._rounds, data)
            return [pickle.loads(value) for value in values]

    async def send(self, obj: Any, dest: int, tag: int) -> None:
        data = _dumps(obj)
        self.sent_bytes += len(data)
        self._world._send(self.rank, dest, tag, data)
        await asyncio.sleep(0)

    async def recv(self, tag: int) -> tuple[int, Any]:
        source, data = await self._world._inbox[self.rank][tag].get()
        return source, pickle.loads(data)


async def run_local(
    size: int,
    rank_main: Callable[[LocalComm], Awaitable[T]],
    chaos: int | None = None,
) -> list[T]:
    """Run rank_main on every rank of a LocalWorld; if one rank fails, cancel the rest."""
    world = LocalWorld(size, chaos)
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(rank_main(comm)) for comm in world.comms]
    return [task.result() for task in tasks]
