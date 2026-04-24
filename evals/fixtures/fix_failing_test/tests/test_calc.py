from calc import sum_range


def test_sum_range_basic():
    assert sum_range(1, 5) == 15


def test_sum_range_single_element():
    assert sum_range(3, 3) == 3


def test_sum_range_zero():
    assert sum_range(0, 0) == 0


def test_sum_range_spans_negative_and_positive():
    assert sum_range(-2, 2) == 0
