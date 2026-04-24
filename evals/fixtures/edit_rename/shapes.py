"""Geometry helpers used across the project. The obsolete name
`compute_area` will be renamed to `calculate_area` as part of an eval
task — do not preserve backwards-compatible aliases; the point of the
task is a clean rename."""
import math


def compute_area(radius: float) -> float:
    """Return the area of a circle with the given radius."""
    return math.pi * radius * radius
