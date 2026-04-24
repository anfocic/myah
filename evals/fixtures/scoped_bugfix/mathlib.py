"""Small math helpers. Contains an off-by-one bug in `sum_range` — the
docstring says inclusive, the implementation is exclusive. Fix the
implementation to match the contract; do not edit the tests."""


def sum_range(start: int, end: int) -> int:
    """Return the sum of integers from start to end, INCLUSIVE of both.

    E.g. sum_range(1, 5) == 1 + 2 + 3 + 4 + 5 == 15.
    """
    return sum(range(start, end))


def is_even(n: int) -> bool:
    return n % 2 == 0
