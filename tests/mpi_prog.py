"""Programs for tests/test_mpi.py to run under mpiexec. Not a test module itself.

    mpiexec -n 2 python tests/mpi_prog.py comm
    mpiexec -n 2 python tests/mpi_prog.py fail
"""
import asyncio
import sys
import time

from parabeam.dist.runtime import MpiComm, abort_on_error


async def comm_checks() -> None:
    comm = MpiComm()
    rank, size = comm.rank, comm.size

    # Allgather: rank order, uneven and empty payloads, a few MB.
    payload = [rank] * (rank * 1000)
    assert await comm.allgather(payload) == [[r] * (r * 1000) for r in range(size)]
    big = bytes([rank]) * 5_000_000
    assert [len(b) for b in await comm.allgather(big)] == [5_000_000] * size

    # Point to point: a ring, then everyone to rank 0 on two tags.
    await comm.send(("ring", rank), (rank + 1) % size, tag=7)
    source, (word, value) = await comm.recv(7)
    assert (word, source, value) == ("ring", (rank - 1) % size, (rank - 1) % size)
    if rank != 0:
        await asyncio.gather(comm.send(rank, 0, tag=8), comm.send(-rank, 0, tag=9))
    else:
        eights = sorted([(await comm.recv(8)) for _ in range(size - 1)])
        nines = sorted([(await comm.recv(9)) for _ in range(size - 1)])
        assert eights == [(r, r) for r in range(1, size)]
        assert nines == [(r, -r) for r in range(1, size)]

    # A rank waiting in a collective keeps its event loop running.
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.001)

    ticker = asyncio.ensure_future(tick())
    if rank == 0:
        time.sleep(0.5)  # blocks rank 0 only, so the others wait for it
    began = time.perf_counter()
    await comm.allgather(None)
    waited = time.perf_counter() - began
    ticker.cancel()
    if rank != 0:
        assert waited > 0.3 and ticks > 50, (waited, ticks)

    # Only rank 0 tries a second collective, while the others have not yet
    # joined the first; on a late rank the first could already be complete.
    if rank == 0:
        first = asyncio.ensure_future(comm.allgather(1))
        await asyncio.sleep(0)  # let it start
        try:
            await comm.allgather(2)
        except RuntimeError as error:
            assert "already running" in str(error)
        else:
            raise AssertionError("two collectives at once must fail")
        assert await first == [1] * size
    else:
        await asyncio.sleep(0.2)
        assert await comm.allgather(1) == [1] * size
    await comm.close()
    print(f"rank {rank} of {size}: OK", flush=True)


async def fail() -> None:
    comm = MpiComm()
    if comm.rank == 1:
        raise ValueError("rank 1 fails on purpose")
    await comm.allgather(None)  # never completes; abort_on_error must end the job


if __name__ == "__main__":
    with abort_on_error():
        asyncio.run({"comm": comm_checks, "fail": fail}[sys.argv[1]]())
