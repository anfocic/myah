"""Config loader tests: merge logic, caching, provenance, edge cases."""
import json
from pathlib import Path

from repl import config_loader
from repl.config_loader import (
    DEFAULTS,
    _deep_merge,
    _load_layer,
    config_paths,
    get_config,
    get_provenance,
    reload_config,
)


class TestDeepMerge:
    def test_merge_adds_new_keys(self):
        base = {"a": 1}
        over = {"b": 2}
        assert _deep_merge(base, over) == {"a": 1, "b": 2}

    def test_merge_overrides_scalars(self):
        base = {"a": 1}
        over = {"a": 99}
        assert _deep_merge(base, over) == {"a": 99}

    def test_merge_recurse_into_dicts(self):
        base = {"ctx": {"num": 10, "keep": True}}
        over = {"ctx": {"num": 20}}
        assert _deep_merge(base, over) == {"ctx": {"num": 20, "keep": True}}

    def test_merge_replaces_dict_with_scalar(self):
        base = {"ctx": {"num": 10}}
        over = {"ctx": 5}
        assert _deep_merge(base, over) == {"ctx": 5}

    def test_merge_does_not_mutate_inputs(self):
        base = {"a": 1}
        over = {"b": 2}
        _deep_merge(base, over)
        assert base == {"a": 1}
        assert over == {"b": 2}


class TestLoadLayer:
    def test_missing_file_returns_none(self, tmp_path):
        assert _load_layer(tmp_path / "nope.json") is None

    def test_valid_json_returns_dict(self, tmp_path):
        p = tmp_path / "cfg.json"
        p.write_text('{"num_ctx": 4096}')
        assert _load_layer(p) == {"num_ctx": 4096}

    def test_invalid_json_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        assert _load_layer(p) is None

    def test_non_dict_json_returns_none(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]")
        assert _load_layer(p) is None


class TestLoadConfig:
    def test_no_files_uses_defaults(self, tmp_path, monkeypatch):
        # Point all paths to a temp dir with no files.
        monkeypatch.setattr(config_loader, "USER_CONFIG", tmp_path / "user.json")
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", tmp_path / "proj.json")
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        cfg = get_config()
        assert cfg["context"]["num_ctx"] == DEFAULTS["context"]["num_ctx"]

    def test_user_layer_overrides_defaults(self, tmp_path, monkeypatch):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"context": {"num_ctx": 4096}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", user)
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", tmp_path / "proj.json")
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        assert get_config()["context"]["num_ctx"] == 4096

    def test_project_layer_overrides_user(self, tmp_path, monkeypatch):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"context": {"num_ctx": 4096}}))
        proj = tmp_path / "proj.json"
        proj.write_text(json.dumps({"context": {"num_ctx": 8192}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", user)
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", proj)
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        assert get_config()["context"]["num_ctx"] == 8192

    def test_local_layer_overrides_project(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj.json"
        proj.write_text(json.dumps({"behavior": {"stream_delay_ms": 0}}))
        local = tmp_path / "local.json"
        local.write_text(json.dumps({"behavior": {"stream_delay_ms": 50}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", tmp_path / "user.json")
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", proj)
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", local)
        reload_config()
        assert get_config()["behavior"]["stream_delay_ms"] == 50

    def test_deep_merge_preserves_untouched_keys(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj.json"
        proj.write_text(json.dumps({"context": {"num_ctx": 4096}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", tmp_path / "user.json")
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", proj)
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        cfg = get_config()
        # Reserved tokens untouched
        assert cfg["context"]["reserved_completion_tokens"] == DEFAULTS["context"]["reserved_completion_tokens"]

    def test_provenance_tracks_sources(self, tmp_path, monkeypatch):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"context": {"num_ctx": 4096}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", user)
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", tmp_path / "proj.json")
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        prov = get_provenance()
        assert prov["context.num_ctx"] == "user"
        assert prov["context.reserved_completion_tokens"] == "default"

    def test_cache_avoids_re_read(self, tmp_path, monkeypatch):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"context": {"num_ctx": 4096}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", user)
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", tmp_path / "proj.json")
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        first = get_config()
        # Mutate file on disk — cache should NOT see it.
        user.write_text(json.dumps({"context": {"num_ctx": 9999}}))
        second = get_config()
        assert first is second
        assert second["context"]["num_ctx"] == 4096

    def test_reload_invalidates_cache(self, tmp_path, monkeypatch):
        user = tmp_path / "user.json"
        user.write_text(json.dumps({"context": {"num_ctx": 4096}}))
        monkeypatch.setattr(config_loader, "USER_CONFIG", user)
        monkeypatch.setattr(config_loader, "PROJECT_CONFIG", tmp_path / "proj.json")
        monkeypatch.setattr(config_loader, "PROJECT_LOCAL_CONFIG", tmp_path / "local.json")
        reload_config()
        user.write_text(json.dumps({"context": {"num_ctx": 9999}}))
        reload_config()
        assert get_config()["context"]["num_ctx"] == 9999


class TestConfigPaths:
    def test_returns_three_paths(self):
        paths = config_paths()
        assert set(paths.keys()) == {"user", "project", "project-local"}

    def test_paths_are_path_objects(self):
        for p in config_paths().values():
            assert isinstance(p, Path)
