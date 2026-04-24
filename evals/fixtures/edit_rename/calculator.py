"""Thin orchestration layer that pipes radii from input to area output.
Imports the geometry helper by name, so renaming the helper means
updating this caller too."""
from shapes import compute_area


def total_area(radii: list[float]) -> float:
    return sum(compute_area(r) for r in radii)


def average_area(radii: list[float]) -> float:
    if not radii:
        return 0.0
    return total_area(radii) / len(radii)
