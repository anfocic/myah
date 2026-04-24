"""Pre-fix: both division cases fail because calc.divide returns a+b
instead of a/b. Multiplication passes regardless — sanity check that
the bug is scoped."""
from calc import divide, multiply


def test_divide_integer_result():
    assert divide(10, 2) == 5


def test_divide_float_result():
    assert divide(1, 4) == 0.25


def test_multiply_unaffected():
    assert multiply(3, 4) == 12
