"""Tests pin the inclusive-range contract described in mathlib.sum_range's
docstring. These must NOT be modified by the task; the fix belongs in
mathlib.py."""
from mathlib import is_even, sum_range


def test_sum_range_small_inclusive():
    # 1 + 2 + 3 + 4 + 5 = 15 — fails pre-fix because range() excludes 5.
    assert sum_range(1, 5) == 15


def test_sum_range_single_point():
    assert sum_range(3, 3) == 3


def test_sum_range_descending_empty():
    # Per the docstring the inclusive range is empty when end < start.
    assert sum_range(5, 1) == 0


def test_is_even_unaffected():
    # Sanity check that the unrelated helper isn't touched.
    assert is_even(4)
    assert not is_even(5)
