"""Geometry helpers used across the project."""
import math


def compute_area(radius: float) -> float:
    """Return the area of a circle with the given radius."""
    return math.pi * radius * radius
