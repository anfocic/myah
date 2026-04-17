"""git_checkout tool guards — the interesting case is the dash-prefix guard
since git would otherwise parse `-f` / `--force` / `--orphan` as flags."""
from tools.git import git_checkout


def test_rejects_dash_prefix_branch():
    result = git_checkout("-f")
    assert result.startswith("Refusing")
    assert "-f" in result


def test_rejects_double_dash_prefix_branch():
    result = git_checkout("--force")
    assert result.startswith("Refusing")


def test_rejects_empty_branch():
    result = git_checkout("")
    assert result.startswith("Refusing")
