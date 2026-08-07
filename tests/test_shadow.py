import os
import shutil
import tempfile
from pathlib import Path

import pytest

from agent.sandbox.shadow import (
    ShadowManager,
    get_active_workspace,
    set_active_shadow,
    clear_active_shadow,
    get_shadow_path,
    create_shadow_if_needed,
    SKIP_DIRS,
    MAX_SHADOW_BYTES,
    _long_path,
)


@pytest.fixture
def real_ws(tmp_path):
    ws = tmp_path / "real_ws"
    ws.mkdir()
    (ws / "file1.py").write_text("print('hello')", encoding="utf-8")
    (ws / "file2.txt").write_text("text content", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "nested.py").write_text("x = 1", encoding="utf-8")
    for d in SKIP_DIRS:
        (ws / d).mkdir()
        (ws / d / "should_skip.py").write_text("skip me", encoding="utf-8")
    (ws / ".gitignore").write_text("*.log\nsecrets/\n", encoding="utf-8")
    (ws / "app.log").write_text("log line", encoding="utf-8")
    (ws / "secrets").mkdir()
    (ws / "secrets" / "key.txt").write_text("secret", encoding="utf-8")
    return str(ws)


@pytest.fixture
def shadow_dir(tmp_path):
    d = tmp_path / "shadow"
    return str(d)


class TestCreateShadow:
    def test_basic_copy(self, real_ws, shadow_dir):
        result = ShadowManager.create_shadow(real_ws, shadow_dir)
        assert result["files"] >= 3
        assert (Path(shadow_dir) / "file1.py").exists()
        assert (Path(shadow_dir) / "file2.txt").exists()
        assert (Path(shadow_dir) / "sub" / "nested.py").exists()

    def test_skip_dirs(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        for d in SKIP_DIRS:
            assert not (Path(shadow_dir) / d).exists(), f"{d} should be skipped"

    def test_gitignore_respected(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        assert not (Path(shadow_dir) / "app.log").exists()
        assert not (Path(shadow_dir) / "secrets").exists()

    def test_hidden_files_skipped(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        assert not (Path(shadow_dir) / ".gitignore").exists()

    def test_skipped_list_returned(self, real_ws, shadow_dir):
        result = ShadowManager.create_shadow(real_ws, shadow_dir)
        assert "app.log" in result["skipped"]

    def test_200mb_limit(self, real_ws, shadow_dir, monkeypatch):
        monkeypatch.setenv("SHADOW_MAX_BYTES", "10")
        with pytest.raises(ValueError, match="超过上限"):
            ShadowManager.create_shadow(real_ws, shadow_dir)

    def test_nonexistent_source(self, shadow_dir):
        with pytest.raises(ValueError):
            ShadowManager.create_shadow("/nonexistent/path", shadow_dir)

    def test_deep_nested_paths(self, tmp_path, shadow_dir):
        """深嵌套路径不应触发 WinError 206（MAX_PATH 限制）。"""
        deep = tmp_path / "real_ws_deep"
        deep.mkdir()
        parts = ["a"] * 20 + ["very_long_directory_name_for_testing_purposes"]
        nested = deep
        for p in parts:
            nested = nested / p
        nested.mkdir(parents=True)
        (nested / "deep_file.py").write_text("x = 1", encoding="utf-8")
        result = ShadowManager.create_shadow(str(deep), shadow_dir)
        assert result["files"] >= 1
        assert (Path(shadow_dir) / "a" / "a").exists()

    def test_deep_nested_diff_and_apply(self, tmp_path, shadow_dir):
        """深嵌套路径的 diff 和 apply 也应正常工作。"""
        deep = tmp_path / "real_ws_deep2"
        deep.mkdir()
        parts = ["lvl1", "lvl2", "lvl3", "lvl4", "lvl5", "lvl6",
                 "lvl7", "lvl8", "lvl9", "lvl10", "lvl11", "lvl12"]
        nested = deep
        for p in parts:
            nested = nested / p
        nested.mkdir(parents=True)
        (nested / "target.py").write_text("val = 1", encoding="utf-8")
        ShadowManager.create_shadow(str(deep), shadow_dir)
        diff = ShadowManager.list_shadow_diff(shadow_dir, str(deep))
        assert diff["added"] == []
        assert diff["modified"] == []
        assert diff["deleted"] == []
        (Path(shadow_dir) / "lvl1" / "lvl2" / "lvl3" / "lvl4" / "lvl5" /
         "lvl6" / "lvl7" / "lvl8" / "lvl9" / "lvl10" / "lvl11" /
         "lvl12" / "target.py").write_text("val = 2", encoding="utf-8")
        diff = ShadowManager.list_shadow_diff(shadow_dir, str(deep))
        rel = os.path.join(*parts, "target.py")
        assert rel in diff["modified"]
        result = ShadowManager.apply_shadow_to_real(shadow_dir, str(deep))
        assert result["synced"] >= 1


class TestLongPath:
    def test_long_path_windows_prefix(self):
        """_long_path 在 Windows 下添加 \\\\?\\ 前缀。"""
        p = _long_path("C:\\Users\\test\\workspace")
        if os.name == "nt":
            assert p.startswith("\\\\?\\")
        else:
            assert p == "C:\\Users\\test\\workspace"

    def test_long_path_idempotent(self):
        """重复应用 _long_path 不应叠加前缀。"""
        p1 = _long_path("C:\\Users\\test")
        p2 = _long_path(p1)
        assert p1 == p2

    def test_long_path_makes_absolute(self):
        """_long_path 返回绝对路径。"""
        p = _long_path(".")
        assert os.path.isabs(p) or p.startswith("\\\\?\\")


class TestListShadowDiff:
    def test_no_changes(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        diff = ShadowManager.list_shadow_diff(shadow_dir, real_ws)
        assert diff["added"] == []
        assert diff["modified"] == []
        assert diff["deleted"] == []

    def test_added_file(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        (Path(shadow_dir) / "new_file.py").write_text("new", encoding="utf-8")
        diff = ShadowManager.list_shadow_diff(shadow_dir, real_ws)
        assert "new_file.py" in diff["added"]

    def test_modified_file(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        (Path(shadow_dir) / "file1.py").write_text("print('modified')", encoding="utf-8")
        diff = ShadowManager.list_shadow_diff(shadow_dir, real_ws)
        assert "file1.py" in diff["modified"]

    def test_deleted_file(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        (Path(shadow_dir) / "file1.py").unlink()
        diff = ShadowManager.list_shadow_diff(shadow_dir, real_ws)
        assert "file1.py" in diff["deleted"]


class TestApplyShadowToReal:
    def test_apply_added(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        (Path(shadow_dir) / "new_file.py").write_text("new content", encoding="utf-8")
        result = ShadowManager.apply_shadow_to_real(shadow_dir, real_ws)
        assert result["synced"] == 1
        assert (Path(real_ws) / "new_file.py").exists()
        assert (Path(real_ws) / "new_file.py").read_text() == "new content"

    def test_apply_modified(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        (Path(shadow_dir) / "file1.py").write_text("print('changed')", encoding="utf-8")
        result = ShadowManager.apply_shadow_to_real(shadow_dir, real_ws)
        assert result["synced"] == 1
        assert (Path(real_ws) / "file1.py").read_text() == "print('changed')"

    def test_apply_deleted(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        (Path(shadow_dir) / "file1.py").unlink()
        result = ShadowManager.apply_shadow_to_real(shadow_dir, real_ws)
        assert result["synced"] == 1
        assert not (Path(real_ws) / "file1.py").exists()

    def test_apply_no_changes(self, real_ws, shadow_dir):
        ShadowManager.create_shadow(real_ws, shadow_dir)
        result = ShadowManager.apply_shadow_to_real(shadow_dir, real_ws)
        assert result["synced"] == 0


class TestShadowRegistry:
    def test_set_and_get(self):
        set_active_shadow(1, "t1", "/tmp/shadow1")
        assert get_active_workspace(1, "t1") == "/tmp/shadow1"
        clear_active_shadow(1, "t1")

    def test_get_none_when_not_set(self):
        assert get_active_workspace(999, "nonexistent") is None

    def test_get_none_when_tid_none(self):
        assert get_active_workspace(1, None) is None

    def test_clear_removes(self, tmp_path):
        sp = str(tmp_path / "shadow_test")
        Path(sp).mkdir()
        set_active_shadow(1, "t2", sp)
        clear_active_shadow(1, "t2")
        assert get_active_workspace(1, "t2") is None
        assert not Path(sp).exists()

    def test_get_shadow_path(self):
        p = get_shadow_path(1, "abc")
        assert "agent_shadow_1_abc" in p

    def test_create_shadow_if_needed_creates_once(self, real_ws, monkeypatch):
        monkeypatch.setattr(
            "agent.memory.user_memory.get_workspace", lambda uid: real_ws
        )
        sp1 = create_shadow_if_needed(1, "need1")
        assert sp1 is not None
        sp2 = create_shadow_if_needed(1, "need1")
        assert sp1 == sp2
        clear_active_shadow(1, "need1")

    def test_create_shadow_if_needed_no_workspace(self, monkeypatch):
        monkeypatch.setattr(
            "agent.memory.user_memory.get_workspace", lambda uid: None
        )
        assert create_shadow_if_needed(1, "need2") is None
