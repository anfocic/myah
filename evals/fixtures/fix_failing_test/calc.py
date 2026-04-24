def sum_range(start: int, end: int) -> int:
    """Return the sum of integers from start to end, inclusive."""
    total = 0
    for i in range(start, end):
        total += i
    return total
