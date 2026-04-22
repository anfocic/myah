import os

from env import load_dotenv


def test_load_dotenv_sets_missing_vars_from_explicit_file(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "# comment",
                "BRAVE_API_KEY=from-file",
                "OPENAI_API_KEY = \"quoted-value\"",
                "export GOOGLE_API_KEY=with-export",
            ]
        )
    )

    for key in ("BRAVE_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(dotenv)

    assert os.environ["BRAVE_API_KEY"] == "from-file"
    assert os.environ["OPENAI_API_KEY"] == "quoted-value"
    assert os.environ["GOOGLE_API_KEY"] == "with-export"


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("BRAVE_API_KEY=from-file\n")

    monkeypatch.setenv("BRAVE_API_KEY", "already-set")

    load_dotenv(dotenv)

    assert os.environ["BRAVE_API_KEY"] == "already-set"
