"""Smoke tests for the geometry helpers. These exist so the post-rename
check (`pytest -q`) verifies the rename was consistent — if the test
file still references the old name after the edit, the import fails
immediately."""
import math

from calculator import average_area, total_area
from shapes import compute_area


def test_compute_area_unit_circle():
    assert compute_area(1) == math.pi


def test_compute_area_zero_radius():
    assert compute_area(0) == 0


def test_total_area_sums_each_circle():
    assert total_area([1, 2]) == math.pi + 4 * math.pi


def test_average_area_handles_empty():
    assert average_area([]) == 0.0
