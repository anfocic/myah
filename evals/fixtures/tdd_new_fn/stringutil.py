def slugify(text: str) -> str:
    """Return a URL-safe slug derived from ``text``.

    Rules:
    - Result is lowercase.
    - Any run of characters that are not ASCII letters or digits collapses
      into a single '-'.
    - Leading and trailing '-' are stripped.
    - An empty or all-separator input returns ''.
    """
    raise NotImplementedError
