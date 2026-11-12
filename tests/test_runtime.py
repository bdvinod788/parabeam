import asyncio

import pytest

from parabeam.dist.runtime import LocalWorld, run_local


def run(coro):
    return asyncio.run(coro)


def test_allgather_returns_every_rank_in_order():
    async def rank_main(comm):
        return await comm.allgather({"rank": comm.rank})

    assert run(run_local(3, rank_main)) == [[{"rank": r} for r in range(3)]] * 3


def test_each_rank_gets_its_own_copies():
    async def rank_main(comm):
        mine = [comm.rank]
        gathered = await comm.allgather(mine)
        if comm.rank == 0:
            gathered[1].append("changed")
        mine.append("changed too")
        return await comm.allgather(mine), gathered

    results = run(run_local(2, rank_main))
    assert results[0][1] == [[0], [1, "changed"]]
    assert results[1][1] == [[0], [1]]  # rank 0's edit did not reach rank 1
    assert results[0][0] == [[0, "changed too"], [1, "changed too"]]


def test_two_collectives_at_once_fail():
    async def rank_main(comm):
        if comm.rank:  # join late, so rank 0's first collective is still running
            for _ in range(10):
                await asyncio.sleep(0)
            return await comm.allgather(1)
        first = asyncio.ensure_future(comm.allgather(1))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already running"):
            await comm.allgather(2)
        return await first

    assert run(run_local(2, rank_main)) == [[1, 1], [1, 1]]


def test_messages_carry_their_source_and_keep_tags_apart():
    async def rank_main(comm):
        if comm.rank:
            await comm.send(("a", comm.rank), 0, tag=1)
            await comm.send(("b", comm.rank), 0, tag=2)
            return None
        twos = sorted([await comm.recv(2) for _ in range(comm.size - 1)])
        ones = sorted([await comm.recv(1) for _ in range(comm.size - 1)])
        return ones, twos

    ones, twos = run(run_local(4, rank_main))[0]
    assert ones == [(r, ("a", r)) for r in (1, 2, 3)]
    assert twos == [(r, ("b", r)) for r in (1, 2, 3)]


def arrivals(chaos):
    async def rank_main(comm):
        if comm.rank:
            for i in range(5):
                await comm.send(i, 0, tag=1)
            return None
        return [await comm.recv(1) for _ in range(5 * (comm.size - 1))]

    return run(run_local(4, rank_main, chaos))[0]


def test_chaos_reorders_deliveries_but_loses_nothing():
    orders = {tuple(arrivals(seed)) for seed in range(5)}
    assert len(orders) == 5
    assert all(sorted(order) == sorted(arrivals(None)) for order in orders)


def test_chaos_replays_from_its_seed():
    assert arrivals(3) == arrivals(3)


def test_bytes_sent_are_counted():
    async def rank_main(comm):
        await comm.allgather(b"x" * 1000)
        if comm.rank == 0:
            await comm.send(b"y" * 500, 1, tag=1)
        else:
            await comm.recv(1)
        return comm.sent_bytes

    sent = run(run_local(2, rank_main))
    assert 1500 <= sent[0] < 1600 and 1000 <= sent[1] < 1100


def test_a_failing_rank_cancels_the_others_instead_of_hanging():
    async def rank_main(comm):
        if comm.rank == 1:
            raise ValueError("rank 1 fails")
        await comm.allgather(None)  # never completes

    with pytest.raises(ExceptionGroup) as caught:
        run(asyncio.wait_for(run_local(3, rank_main), timeout=5))
    assert caught.group_contains(ValueError, match="rank 1 fails")


def test_sending_to_a_missing_rank_fails():
    async def rank_main(comm):
        await comm.send(1, 5, tag=1)

    with pytest.raises(ExceptionGroup) as caught:
        run(run_local(2, rank_main))
    assert caught.group_contains(ValueError, match="no rank 5")


def test_a_world_needs_a_rank():
    with pytest.raises(ValueError):
        LocalWorld(0)
