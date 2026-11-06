import os
import subprocess
import sys

from parabeam.core.seeding import child_seed, path_key


def test_path_key_format():
    assert path_key(17, (2, 0, 1)) == "p17/L3/2-0-1"
    assert path_key(17, ()) == "p17/L0/"


def test_seeds_are_pinned():
    # Changing these silently changes every result in the report.
    assert child_seed(17, (2, 0), 1) == 2543258962492052237
    assert child_seed(0, (), 0) == 5262934928381767526


def test_seed_depends_on_every_part_of_the_path():
    seeds = {
        child_seed(p, parent, child)
        for p in range(4)
        for parent in [(), (0,), (1,), (0, 0), (0, 1), (1, 0), (10,), (1, 1)]
        for child in range(8)
    }
    assert len(seeds) == 4 * 8 * 8


def test_seed_does_not_confuse_multi_digit_indices():
    assert child_seed(1, (1, 11), 1) != child_seed(1, (11, 1), 1)
    assert child_seed(1, (1,), 11) != child_seed(11, (1,), 1)


def test_seed_fits_vllm_range():
    for child in range(1000):
        assert 0 <= child_seed(3, (1, 2), child) < 2**63


def test_seed_survives_a_new_process_with_a_different_hash_seed():
    code = "from parabeam.core.seeding import child_seed; print(child_seed(17, (2, 0), 1))"
    outputs = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": os.getcwd()}
        outputs.add(subprocess.check_output([sys.executable, "-c", code], env=env))
    assert len(outputs) == 1
