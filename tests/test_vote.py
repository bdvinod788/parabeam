import random

from parabeam.backends.base import StepOut
from parabeam.core.beam import Beam, Problem
from parabeam.core.vote import weighted_vote

PROBLEM = Problem(id=0, gen_prompt_ids=(), prm_prompt_ids=())


def beam(answer, score, index=0):
    out = StepOut([1], answer or "", "eos", 0.0, 0.0)
    return Beam(PROBLEM).child(index, out, [1]).scored(score)


def extract(text):
    return text or None


def exact(a, b):
    return a == b


def test_weight_beats_count():
    beams = [beam("a", 0.9, 0), beam("b", 0.3, 1), beam("b", 0.3, 2)]
    vote = weighted_vote(beams, extract, exact)
    assert (vote.answer, vote.weight, vote.voters) == ("a", 0.9, 3)


def test_many_small_weights_can_win():
    beams = [beam("a", 0.5, 0), beam("b", 0.3, 1), beam("b", 0.3, 2)]
    assert weighted_vote(beams, extract, exact).answer == "b"


def test_equivalent_answers_pool_their_weight():
    same = lambda a, b: {a, b} <= {"1/2", "0.5"}
    beams = [beam("1/2", 0.3, 0), beam("0.5", 0.3, 1), beam("7", 0.5, 2)]
    vote = weighted_vote(beams, extract, same)
    assert vote.weight == 0.6
    assert vote.answer in {"1/2", "0.5"}


def test_answer_comes_from_the_best_beam_in_the_group():
    same = lambda a, b: {a, b} <= {"1/2", "0.5"}
    beams = [beam("1/2", 0.2, 0), beam("0.5", 0.3, 1)]
    vote = weighted_vote(beams, extract, same)
    assert vote.answer == "0.5"
    assert vote.beam.path == (1,)


def test_exact_tie_goes_to_the_answer_that_sorts_first():
    beams = [beam("b", 0.5, 0), beam("a", 0.5, 1)]
    assert weighted_vote(beams, extract, exact).answer == "a"


def test_beams_without_answers_do_not_vote():
    beams = [beam(None, 0.9, 0), beam("a", 0.1, 1)]
    vote = weighted_vote(beams, extract, exact)
    assert (vote.answer, vote.voters) == ("a", 1)


def test_no_answers_at_all():
    assert weighted_vote([beam(None, 0.9)], extract, exact) is None
    assert weighted_vote([], extract, exact) is None


def test_weight_does_not_depend_on_order_even_when_running_totals_do():
    rng = random.Random(0)
    beams = [beam("a", rng.random(), i) for i in range(32)]

    def running_total(items):
        total = 0.0
        for item in items:
            total += item.score
        return total

    weights, totals = set(), set()
    for _ in range(50):
        rng.shuffle(beams)
        weights.add(weighted_vote(beams, extract, exact).weight)
        totals.add(running_total(beams))
    assert len(totals) > 1
    assert len(weights) == 1


def test_grouping_does_not_depend_on_order():
    # "similar" is not transitive, so grouping must follow a fixed order.
    similar = lambda a, b: abs(int(a) - int(b)) <= 1
    beams = [beam(str(i), 0.1 * (i + 1), i) for i in (1, 2, 3, 4, 5)]
    rng = random.Random(0)
    results = set()
    for _ in range(50):
        rng.shuffle(beams)
        results.add(weighted_vote(beams, extract, similar).answer)
    assert len(results) == 1
