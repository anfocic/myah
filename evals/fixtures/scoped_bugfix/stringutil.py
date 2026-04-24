"""Unrelated string helpers. Included in the fixture as a distractor —
the task is to fix `mathlib.py`; this file must remain byte-identical."""


def reverse(s: str) -> str:
    return s[::-1]


def capitalize_words(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())
