"""The mock backend and the cost model it replays."""
import asyncio
import json
import statistics
import time

import pytest

from parabeam.backends.base import Aborted, GenReq
from parabeam.backends.mock import (
    EXTRA_0,
    PROBLEM_STEPS,
    STEP_TOKENS,
    STOP_SPREAD,
    Chaos,
    MockBackend,
    PrefixCache,
    mock_problem,
    mock_score,
    mock_step,
    stop_chance,
)
from parabeam.core.seeding import child_seed, path_key
from parabeam.dist.costmodel import GUESS, CostModel
from parabeam.tools.fit_steps import fit, inverse_cdf


def request(index=0, level=3, problem=17, max_tokens=2048, final=False, prefix=None):
    path = (0,) * (level - 1)
    return GenReq(
        beam_id=path_key(problem, (*path, index)),
        prefix_ids=prefix if prefix is not None else [1, 2, 3],
        seed=child_seed(problem, path, index),
        max_tokens=max_tokens,
        final=final,
    )


def requests(count, **kwargs):
    return [request(index=i, **kwargs) for i in range(count)]


async def gather_all(backend, reqs):
    return await asyncio.gather(*(backend.generate(r) for r in reqs))


def test_outputs_depend_only_on_the_request():
    assert mock_step(request()) == mock_step(request())
    assert mock_step(request(1)) != mock_step(request(2))


def test_step_lengths_follow_prm800k():
    lengths = sorted(len(mock_step(r)[0]) for r in requests(20000, level=1, problem=10**6))
    assert statistics.median(lengths) == pytest.approx(STEP_TOKENS[500], abs=2)
    assert lengths[19800] == pytest.approx(STEP_TOKENS[990], rel=0.15)
    assert statistics.mean(lengths) == pytest.approx(statistics.mean(STEP_TOKENS), rel=0.05)


def test_step_statistics_are_sane():
    assert STEP_TOKENS == sorted(STEP_TOKENS) and PROBLEM_STEPS == sorted(PROBLEM_STEPS)
    assert STEP_TOKENS[0] >= 1 and PROBLEM_STEPS[0] >= 1
    assert STOP_SPREAD >= 1


@pytest.mark.parametrize("problem", range(50))
def test_every_solution_ends_within_the_window(problem):
    chances = [stop_chance(problem, level) for level in range(1, 200)]
    assert all(0.0 <= chance <= 1.0 for chance in chances)
    end = chances.index(1.0) + 1
    assert end >= 1 + STOP_SPREAD
    assert end <= PROBLEM_STEPS[-1] + STOP_SPREAD
    assert chances[:end] == sorted(chances[:end])


def test_no_solution_ends_before_its_window():
    late = 0
    for problem in range(200):
        chances = [stop_chance(problem, level) for level in range(1, 200)]
        first = chances.index(1.0) + 1 - 2 * STOP_SPREAD
        if first > 1:
            late += 1
            assert not any(chances[: first - 1])
            assert chances[first - 1] > 0
    assert late > 20


def test_stopping_is_uniform_over_the_window():
    problem = 17
    alive, ends = 1.0, []
    for level in range(1, 200):
        ends.append(alive * stop_chance(problem, level))
        alive -= ends[-1]
    window = [p for p in ends if p > 0]
    assert STOP_SPREAD <= len(window) <= 2 * STOP_SPREAD + 1
    assert all(p == pytest.approx(window[0]) for p in window)
    assert alive == pytest.approx(0.0)


def test_steps_at_a_level_stop_at_the_stated_rate():
    level = next(lv for lv in range(1, 60) if 0.05 < stop_chance(17, lv) < 0.5)
    stops = sum(mock_step(r)[2] == "eos" for r in requests(4000, level=level))
    assert stops / 4000 == pytest.approx(stop_chance(17, level), abs=0.03)


def test_finish_reasons_match_their_text():
    for r in requests(500, level=20):
        tokens, text, finish = mock_step(r)
        assert tokens
        assert text.endswith("\n\n") == (finish == "stop")
        assert ("\\boxed" in text) == (finish == "eos")


def test_the_cap_cuts_a_step_and_drops_its_answer():
    tokens, text, finish = mock_step(request(max_tokens=2, level=1))
    assert len(tokens) == 2 and finish == "length" and "\\boxed" not in text


def test_final_requests_never_stop_at_a_blank_line():
    finishes = {mock_step(r)[2] for r in requests(200, final=True, level=1)}
    assert finishes == {"eos"}
    capped = {mock_step(r)[2] for r in requests(200, final=True, level=1, max_tokens=20)}
    assert "length" in capped


def test_prm_step_ids_end_in_extra_0():
    out = asyncio.run(MockBackend().generate(request()))
    ids = MockBackend().prm_step_ids(out)
    assert ids == [*out.token_ids, EXTRA_0]
    assert EXTRA_0 not in out.token_ids


def test_scores_ignore_batch_and_order():
    a, b, c = [1, 2], [1, 3], [7] * 300

    async def run():
        backend = MockBackend()
        return await backend.score([a, b, c]), await backend.score([c, a]), await backend.score([b])

    one, two, three = asyncio.run(run())
    assert one == [mock_score(a), mock_score(b), mock_score(c)]
    assert two == [one[2], one[0]] and three == [one[1]]
    assert all(0.0 <= score < 1.0 for score in one)


def test_mock_problems_are_fixed():
    assert mock_problem(5) == mock_problem(5)
    assert mock_problem(5) != mock_problem(6)


def test_chaos_changes_timing_never_content():
    reqs = requests(32)
    plain = asyncio.run(gather_all(MockBackend(), reqs))
    shaken = asyncio.run(gather_all(MockBackend(chaos=Chaos(1)), reqs))
    for a, b in zip(plain, shaken):
        assert (a.token_ids, a.text, a.finish) == (b.token_ids, b.text, b.finish)
    key = lambda pair: pair[1].t_end
    arrival = lambda outs: [i for i, _ in sorted(enumerate(outs), key=key)]
    assert arrival(plain) != arrival(shaken)


def test_a_bigger_batch_is_slower_overall_but_faster_per_request():
    async def elapsed(count):
        backend = MockBackend()
        await gather_all(backend, [r for r in requests(count, max_tokens=8)])
        return backend.now()

    small, large = asyncio.run(elapsed(4)), asyncio.run(elapsed(128))
    assert small < large < small * 128 / 4


def test_requests_run_as_a_continuous_batch():
    async def run():
        backend = MockBackend()
        long, short = request(max_tokens=2048, level=1), request(1, max_tokens=2, level=1)
        return await asyncio.gather(backend.generate(long), backend.generate(short))

    long_out, short_out = asyncio.run(run())
    assert short_out.t_end <= long_out.t_end
    assert long_out.t_start == short_out.t_start


def test_a_cached_prefix_is_not_prefilled_again():
    prefix = list(range(1600))

    async def run(backend, req):
        start = backend.now()
        await backend.generate(req)
        return backend.now() - start

    async def both():
        backend = MockBackend()
        cold = await run(backend, request(0, prefix=prefix))
        warm = await run(backend, request(0, level=4, prefix=prefix))
        return cold, warm

    cold, warm = asyncio.run(both())
    assert cold - warm == pytest.approx(GUESS.prefill(1600), rel=0.5)


def test_scoring_charges_only_new_tokens():
    async def run():
        backend = MockBackend()
        await backend.score([list(range(1000))])
        first = backend.now()
        await backend.score([list(range(1000))])
        return first, backend.now() - first

    first, repeat = asyncio.run(run())
    assert first == pytest.approx(1000 * GUESS.prm_s_per_token)
    assert repeat == pytest.approx(8 * GUESS.prm_s_per_token)  # past the last whole block


def test_prefix_cache_matches_whole_leading_blocks():
    cache = PrefixCache()
    ids = list(range(100))
    assert cache.cached(ids) == 0
    cache.add(ids)
    assert cache.cached(ids) == 96
    assert cache.cached(ids[:40]) == 32
    assert cache.cached([*ids[:40], 999, *ids[41:]]) == 32
    assert cache.cached([999, *ids[1:]]) == 0


def test_prefix_cache_evicts_the_oldest_blocks():
    cache = PrefixCache(capacity=2)
    cache.add(list(range(32)))
    cache.add(list(range(100, 116)))
    assert cache.cached(list(range(32))) == 0
    assert cache.cached(list(range(100, 116))) == 16


def test_abort_ends_a_running_request():
    async def run():
        backend = MockBackend()
        req = request(max_tokens=2048, level=1)
        task = asyncio.create_task(backend.generate(req))
        await asyncio.sleep(0)
        backend.abort(req.beam_id)
        with pytest.raises(Aborted):
            await task
        again = await backend.generate(req)  # the slot is free
        return again

    assert asyncio.run(run()).token_ids


def test_abort_reaches_a_request_held_by_chaos():
    async def run():
        backend = MockBackend(chaos=Chaos(0, max_hold=1000, stall_chance=1.0))
        reqs = requests(8)
        tasks = [asyncio.create_task(backend.generate(r)) for r in reqs]
        await asyncio.sleep(0)
        for r in reqs:
            backend.abort(r.beam_id)
        return await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(result, Aborted) for result in asyncio.run(run()))


def test_aborting_a_finished_or_unknown_request_does_nothing():
    async def run():
        backend = MockBackend()
        req = request()
        out = await backend.generate(req)
        backend.abort(req.beam_id)
        backend.abort("p0/L1/9")
        return out

    assert asyncio.run(run()).token_ids


def test_an_aborted_request_stops_costing_time():
    async def run(abort):
        backend = MockBackend()
        long, short = request(0, max_tokens=4, level=1), request(1, max_tokens=4, level=1)
        tasks = [asyncio.create_task(backend.generate(r)) for r in (long, short)]
        await asyncio.sleep(0)
        if abort:
            backend.abort(long.beam_id)
        await asyncio.gather(*tasks, return_exceptions=True)
        return backend.now()

    assert asyncio.run(run(True)) < asyncio.run(run(False))


def test_a_running_duplicate_is_refused():
    async def run():
        backend = MockBackend()
        req = request()
        task = asyncio.create_task(backend.generate(req))
        await asyncio.sleep(0)
        with pytest.raises(ValueError):
            await backend.generate(req)
        await task

    asyncio.run(run())


def test_cancelling_a_search_leaves_no_task_running():
    async def run():
        backend = MockBackend()
        task = asyncio.create_task(backend.generate(request(max_tokens=2048, level=1)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return backend._jobs

    assert asyncio.run(run()) == {}


def test_scale_sleeps_for_real():
    async def run():
        backend = MockBackend(scale=0.2)
        start = time.perf_counter()
        await backend.score([list(range(1000))])
        return time.perf_counter() - start, backend.now()

    wall, simulated = asyncio.run(run())
    assert wall >= 1000 * GUESS.prm_s_per_token * 0.2 * 0.9
    assert simulated == pytest.approx(wall / 0.2, rel=0.5)


def test_decode_interpolates_between_measurements():
    cost = CostModel((500, 1000), (1, 3), ((1.0, 2.0), (3.0, 4.0)), (0.5, 1.0), 0.0)
    assert cost.decode(1, 500) == 1.0
    assert cost.decode(3, 1000) == 4.0
    assert cost.decode(2, 750) == pytest.approx(2.5)


def test_costs_beyond_the_measurements():
    cost = CostModel((500, 1000), (1, 3), ((1.0, 2.0), (3.0, 4.0)), (0.5, 1.0), 0.0)
    assert cost.decode(5, 500) == 3.0  # a larger batch keeps getting slower
    assert cost.decode(1, 100) == 1.0
    assert cost.prefill(0) == 0.0
    assert cost.prefill(250) == 0.25
    assert cost.prefill(2000) == 2.0


def test_the_guess_is_monotonic():
    times = [GUESS.decode(batch, 1000) for batch in (1, 8, 64, 256, 512)]
    assert times == sorted(times)


def test_cost_model_loads_what_calibrate_will_write(tmp_path):
    path = tmp_path / "cost_model.json"
    path.write_text(
        json.dumps(
            {
                "contexts": list(GUESS.contexts),
                "batches": list(GUESS.batches),
                "decode_s": [list(row) for row in GUESS.decode_s],
                "prefill_s": list(GUESS.prefill_s),
                "prm_s_per_token": GUESS.prm_s_per_token,
            }
        )
    )
    assert CostModel.load(path) == GUESS


@pytest.mark.parametrize("bad", [((1.0,),), ((1.0, 2.0), (3.0,))])
def test_cost_model_rejects_wrong_shapes(bad):
    with pytest.raises(ValueError):
        CostModel((500, 1000), (1, 3), bad, (0.5, 1.0), 0.0)


def test_fit_summarises_steps_within_and_across_problems():
    stats = fit(
        [1, 2, 3, 100],
        {"short": [4, 4, 4, 4, 4], "long": [10, 12, 14, 10, 14], "thin": [99]},
    )
    assert stats["step_tokens"][0] == 1 and stats["step_tokens"][-1] == 100
    assert stats["problem_steps"][0] == 4 and stats["problem_steps"][-1] == 12
    assert stats["problems"] == 2 and stats["solutions"] == 11
    assert stats["within_problem_sd"] == pytest.approx((3.2 / 2) ** 0.5)


def test_inverse_cdf_covers_the_range():
    table = inverse_cdf([1, 2, 3])
    assert table[0] == 1 and table[-1] == 3 and table == sorted(table)


def flat_cost(decode=0.0, prefill=0.0, prm=0.0, per_context=0.0):
    """A cost model with one lever at a time, so each test sees one effect."""
    contexts, batches = (1000, 2000), (1, 256)
    return CostModel(
        contexts,
        batches,
        tuple((decode + per_context * c,) * 2 for c in contexts),
        (prefill * 1000, prefill * 2000),
        prm,
    )


def test_a_step_extends_its_own_context():
    async def run():
        backend = MockBackend(cost=flat_cost(per_context=1.0))
        out = await backend.generate(request(level=1, max_tokens=20, prefix=[1] * 1000))
        return backend.now(), len(out.token_ids)

    elapsed, tokens = asyncio.run(run())
    assert elapsed == pytest.approx(sum(1000 + i for i in range(tokens)))


def test_requests_sharing_a_prefix_prefill_it_once():
    async def run():
        backend = MockBackend(cost=flat_cost(prefill=1.0))
        reqs = [request(i, level=1, max_tokens=2, prefix=list(range(1600))) for i in range(4)]
        await gather_all(backend, reqs)
        return backend.now()

    assert asyncio.run(run()) == pytest.approx(1600)


def test_a_child_finds_its_parents_output_in_the_cache():
    prefix = list(range(1600))

    async def run():
        backend = MockBackend(cost=flat_cost(prefill=1.0))
        out = await backend.generate(request(0, level=1, prefix=prefix))
        before = backend.now()
        child = request(0, level=2, max_tokens=2, prefix=prefix + out.token_ids)
        await backend.generate(child)
        return backend.now() - before

    assert asyncio.run(run()) < 16 * 2  # only the unfilled tail of the last block


def test_abort_during_a_tick_leaves_the_others_running():
    async def run():
        backend = MockBackend()
        doomed = request(0, level=1, max_tokens=1)
        other = request(1, level=1, max_tokens=3)
        tasks = [asyncio.create_task(backend.generate(r)) for r in (doomed, other)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        backend.abort(doomed.beam_id)
        return await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)

    aborted, finished = asyncio.run(run())
    assert isinstance(aborted, Aborted)
    assert finished.token_ids


def test_a_request_aborted_before_it_joined_costs_nothing():
    async def run():
        backend = MockBackend(chaos=Chaos(0, max_hold=1000, stall_chance=1.0))
        req = requests(1, prefix=list(range(1600)))[0]
        task = asyncio.create_task(backend.generate(req))
        await asyncio.sleep(0)
        backend.abort(req.beam_id)
        await asyncio.gather(task, return_exceptions=True)
        return backend._prefill_s, backend._gen_cache.cached(req.prefix_ids)

    assert asyncio.run(run()) == (0.0, 0)


def test_the_prm_server_scores_one_batch_at_a_time():
    async def run():
        backend = MockBackend(cost=flat_cost(prm=0.0001), scale=1.0)
        start = time.perf_counter()
        await asyncio.gather(backend.score([[1] * 200]), backend.score([[2] * 200]))
        return time.perf_counter() - start

    assert asyncio.run(run()) >= 0.039


def test_chaos_makes_ticks_uneven():
    async def run(chaos):
        backend = MockBackend(chaos=chaos)
        await gather_all(backend, requests(8))
        return backend.now()

    assert asyncio.run(run(Chaos(1, max_hold=1, stall_chance=0.0))) > asyncio.run(run(None))


def test_the_chaos_seed_sets_the_arrival_order():
    def arrival(seed):
        outs = asyncio.run(gather_all(MockBackend(chaos=Chaos(seed, jitter=0.0)), requests(32)))
        return sorted(range(len(outs)), key=lambda i: (outs[i].t_end, i))

    assert arrival(1) != arrival(2)
    assert arrival(1) == arrival(1)


def test_stalls_hold_requests_longer():
    async def turns(stall_chance):
        backend = MockBackend(chaos=Chaos(5, max_hold=16, stall_chance=stall_chance))
        hold = asyncio.create_task(backend._hold("key"))
        count = 0
        while not hold.done():
            await asyncio.sleep(0)
            count += 1
        return count

    assert asyncio.run(turns(1.0)) > asyncio.run(turns(0.0))


def test_prefix_cache_keeps_what_was_used_recently():
    cache = PrefixCache(capacity=2)
    old, new = list(range(16)), list(range(100, 116))
    cache.add(old)
    cache.add(new)
    cache.add(old)
    cache.add(list(range(200, 216)))
    assert cache.cached(old) == 16
    assert cache.cached(new) == 0
