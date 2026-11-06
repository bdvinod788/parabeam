import pytest

from parabeam.eval.grade import is_correct, last_boxed, same_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"so \boxed{5}.", "5"),
        (r"\boxed{1} then \boxed{2}", "2"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"$\boxed{ 7 }$", "7"),
        (r"\boxed{\boxed{3}}", "3"),
        (r"\boxed{\text{a {b} c}}", r"\text{a {b} c}"),
        ("the answer is 5", None),
        ("", None),
        (r"\boxed{}", None),
        (r"\boxed{\frac{1}{2", None),
        (r"\boxed{4} and then \boxed{\frac{1", "4"),
        (r"\boxed{4} then \boxed{}", "4"),
    ],
)
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize(
    "a, b",
    [
        (r"\frac{1}{2}", "0.5"),
        ("72", "72.0"),
        ("1000", "1,000"),
        (r"\sqrt{12}", r"2\sqrt{3}"),
        ("x^2+1", "1+x^2"),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})"),
        (r"\text{Evelyn}", "Evelyn"),
    ],
)
def test_same_answer(a, b):
    assert same_answer(a, b)
    assert same_answer(b, a)


@pytest.mark.parametrize("a, b", [("5", "4"), (r"\frac{1}{2}", "0.25"), ("Evelyn", "Bob")])
def test_different_answers(a, b):
    assert not same_answer(a, b)


def test_is_correct():
    assert is_correct("0.5", r"\frac{1}{2}")
    assert not is_correct("3", r"\frac{1}{2}")
    assert not is_correct(None, "5")
