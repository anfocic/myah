from stringutil import slugify


def test_basic_words():
    assert slugify("Hello World") == "hello-world"


def test_strips_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_collapses_runs():
    assert slugify("a--b  c") == "a-b-c"


def test_strips_edges():
    assert slugify("  --hi--  ") == "hi"


def test_empty_string():
    assert slugify("") == ""


def test_only_separators():
    assert slugify("---") == ""


def test_non_ascii_dropped():
    assert slugify("café au lait") == "caf-au-lait"


def test_digits_preserved():
    assert slugify("Top 10 Things") == "top-10-things"
