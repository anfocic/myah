"""Tiny calculator module with one deliberate bug — `divide` returns
the sum instead of the quotient. Fixing it is turn 1 of the eval task;
turn 2 asks the model to add a new divide-by-zero test."""


def divide(a: float, b: float) -> float:
    """Return a / b."""
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b
