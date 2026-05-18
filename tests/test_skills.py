"""Skills loader — discovers user-defined skill markdown files under
~/.mia/skills/ (or MIA_SKILLS_PATH) and renders them as a `<skills>`
block in the system prompt so the model has the user's per-project
playbook on every turn without burning a tool call."""
from pathlib import Path

import pytest

from agent.skills import Skill, load_skills, skills_block


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


def test_load_skills_returns_empty_when_dir_missing(tmp_path: Path):
    """A missing skills directory must not raise — the feature is opt-in,
    and most users won't have one configured at first."""
    assert load_skills(tmp_path / "nonexistent") == []


def test_load_skills_returns_empty_when_dir_has_no_markdown(skills_dir: Path):
    (skills_dir / "notes.txt").write_text("ignored")
    (skills_dir / "README").write_text("ignored")
    assert load_skills(skills_dir) == []


def test_load_skills_with_frontmatter(skills_dir: Path):
    (skills_dir / "git-flow.md").write_text(
        "---\n"
        "name: git-flow\n"
        "description: Use squash merges; rebase before push\n"
        "---\n\n"
        "Body text describing the git workflow.\n"
    )
    skills = load_skills(skills_dir)
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "git-flow"
    assert s.description == "Use squash merges; rebase before push"
    assert "Body text describing the git workflow." in s.body


def test_load_skills_without_frontmatter_falls_back_to_filename_and_first_line(
    skills_dir: Path,
):
    (skills_dir / "writing-tone.md").write_text(
        "Keep prose tight; cut filler words and hedges.\n\nMore detail here.\n"
    )
    skills = load_skills(skills_dir)
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "writing-tone"
    assert s.description == "Keep prose tight; cut filler words and hedges."
    assert "More detail here." in s.body


def test_load_skills_sorted_by_name(skills_dir: Path):
    (skills_dir / "zeta.md").write_text("z body")
    (skills_dir / "alpha.md").write_text("a body")
    (skills_dir / "mike.md").write_text("m body")
    skills = load_skills(skills_dir)
    assert [s.name for s in skills] == ["alpha", "mike", "zeta"]


def test_load_skills_skips_unreadable_files(skills_dir: Path, monkeypatch):
    """A bad skill file (unreadable, malformed frontmatter) is skipped
    silently — one broken file shouldn't take down the whole loader."""
    (skills_dir / "good.md").write_text("good body")
    bad = skills_dir / "bad.md"
    bad.write_text("---\nname: bad\n  : malformed yaml :::\n---\nbody")

    skills = load_skills(skills_dir)
    names = [s.name for s in skills]
    assert "good" in names


def test_skills_block_returns_none_when_empty():
    assert skills_block([]) is None


def test_skills_block_renders_skills_with_name_description_body():
    skills = [
        Skill(name="alpha", description="first", body="alpha body"),
        Skill(name="beta", description="second", body="beta body"),
    ]
    block = skills_block(skills)
    assert block is not None
    assert block.startswith("<skills>")
    assert block.endswith("</skills>")
    assert "alpha" in block
    assert "first" in block
    assert "alpha body" in block
    assert "beta" in block
    assert "second" in block
    assert "beta body" in block


def test_system_prompt_includes_skills_block_when_present(
    monkeypatch, skills_dir: Path,
):
    """`build_system_prompt` should inject the rendered `<skills>` block
    when skills are loaded — re-reading every turn so a freshly added
    skill file takes effect on the next call."""
    from agent import system_prompt as sp

    (skills_dir / "vim.md").write_text(
        "---\nname: vim\ndescription: vim keymap reminders\n---\n\nBody.\n"
    )

    monkeypatch.setattr(sp, "_skills_for_prompt", lambda: load_skills(skills_dir))

    prompt = sp.build_system_prompt(cwd=str(skills_dir))
    assert "<skills>" in prompt
    assert "vim" in prompt
    assert "vim keymap reminders" in prompt


def test_system_prompt_omits_skills_block_when_none(monkeypatch):
    from agent import system_prompt as sp

    monkeypatch.setattr(sp, "_skills_for_prompt", lambda: [])

    prompt = sp.build_system_prompt(cwd="/tmp")
    assert "<skills>" not in prompt


def test_subagent_does_not_inherit_skills(monkeypatch, skills_dir: Path):
    """Subagents run with a fresh focus and should NOT inherit the user's
    skill set — keep them lean and on-task for a single delegated job."""
    from agent import system_prompt as sp

    (skills_dir / "any.md").write_text("any body")
    monkeypatch.setattr(sp, "_skills_for_prompt", lambda: load_skills(skills_dir))

    prompt = sp.build_system_prompt(cwd=str(skills_dir), subagent=True)
    assert "<skills>" not in prompt
