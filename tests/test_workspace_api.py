import os
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from agent.api import app, get_current_user
from agent.sandbox.shadow import (
    ShadowManager,
    set_active_shadow,
    clear_active_shadow,
    get_shadow_path,
)
from agent.sandbox.snapshot import init_snapshots_table


def _fake_user():
    return {"id": 1, "username": "tester"}


class FakeSettings:
    sqlite_path = ""


def _setup_workspace(tmp_path, monkeypatch):
    real_ws = tmp_path / "real_ws"
    real_ws.mkdir()
    (real_ws / "file1.py").write_text("original", encoding="utf-8")
    (real_ws / "file2.py").write_text("original2", encoding="utf-8")

    db_path = str(tmp_path / "test_ws_api.sqlite")
    init_snapshots_table(db_path)
    FakeSettings.sqlite_path = db_path
    monkeypatch.setattr("agent.api.Settings", lambda: FakeSettings())
    monkeypatch.setattr("agent.api.get_workspace", lambda uid: str(real_ws))
    return str(real_ws)


class TestGetDiff:
    def test_diff_no_shadow(self, tmp_path, monkeypatch):
        _setup_workspace(tmp_path, monkeypatch)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.get("/workspace/diff", params={"thread_id": "t1"})
                assert r.status_code == 200
                data = r.json()
                assert data["pending_sync"] is False
                assert data["diff"] == {"added": [], "modified": [], "deleted": []}
        finally:
            app.dependency_overrides.clear()

    def test_diff_with_changes(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "t2")
        ShadowManager.create_shadow(real_ws, shadow_path)
        (Path(shadow_path) / "file1.py").write_text("modified", encoding="utf-8")
        (Path(shadow_path) / "new_file.py").write_text("new", encoding="utf-8")
        (Path(shadow_path) / "file2.py").unlink()
        set_active_shadow(1, "t2", shadow_path)

        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.get("/workspace/diff", params={"thread_id": "t2"})
                assert r.status_code == 200
                data = r.json()
                assert data["pending_sync"] is True
                assert "new_file.py" in data["diff"]["added"]
                assert "file1.py" in data["diff"]["modified"]
                assert "file2.py" in data["diff"]["deleted"]
        finally:
            clear_active_shadow(1, "t2")
            app.dependency_overrides.clear()


class TestSyncWorkspace:
    def test_sync_applies_changes(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "t3")
        ShadowManager.create_shadow(real_ws, shadow_path)
        (Path(shadow_path) / "file1.py").write_text("modified", encoding="utf-8")
        (Path(shadow_path) / "new_file.py").write_text("new", encoding="utf-8")
        set_active_shadow(1, "t3", shadow_path)

        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/sync", params={"thread_id": "t3"})
                assert r.status_code == 200
                data = r.json()
                assert data["synced"] == 2
                assert data["snapshot_id"] > 0
                assert (Path(real_ws) / "file1.py").read_text() == "modified"
                assert (Path(real_ws) / "new_file.py").read_text() == "new"
        finally:
            clear_active_shadow(1, "t3")
            app.dependency_overrides.clear()

    def test_sync_no_changes(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "t4")
        ShadowManager.create_shadow(real_ws, shadow_path)
        set_active_shadow(1, "t4", shadow_path)

        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/sync", params={"thread_id": "t4"})
                assert r.status_code == 200
                data = r.json()
                assert data["synced"] == 0
        finally:
            clear_active_shadow(1, "t4")
            app.dependency_overrides.clear()

    def test_sync_no_shadow(self, tmp_path, monkeypatch):
        _setup_workspace(tmp_path, monkeypatch)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/sync", params={"thread_id": "t5"})
                assert r.status_code == 400
        finally:
            app.dependency_overrides.clear()


class TestRevertWorkspace:
    def test_revert_restores_files(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "t6")
        ShadowManager.create_shadow(real_ws, shadow_path)
        (Path(shadow_path) / "file1.py").write_text("modified", encoding="utf-8")
        set_active_shadow(1, "t6", shadow_path)

        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/sync", params={"thread_id": "t6"})
                assert r.status_code == 200
                assert (Path(real_ws) / "file1.py").read_text() == "modified"

                r2 = c.post("/workspace/revert", params={"thread_id": "t6"})
                assert r2.status_code == 200
                assert (Path(real_ws) / "file1.py").read_text() == "original"
        finally:
            clear_active_shadow(1, "t6")
            app.dependency_overrides.clear()

    def test_revert_no_snapshot(self, tmp_path, monkeypatch):
        _setup_workspace(tmp_path, monkeypatch)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/revert", params={"thread_id": "t7"})
                assert r.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestVerifyShadow:
    def test_verify_pass(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "v1")
        ShadowManager.create_shadow(real_ws, shadow_path)
        set_active_shadow(1, "v1", shadow_path)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/verify", params={"thread_id": "v1", "command": "echo ok"})
                assert r.status_code == 200
                data = r.json()
                assert data["passed"] is True
                assert "ok" in data["output"]
        finally:
            clear_active_shadow(1, "v1")
            app.dependency_overrides.clear()

    def test_verify_fail(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "v2")
        ShadowManager.create_shadow(real_ws, shadow_path)
        set_active_shadow(1, "v2", shadow_path)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/verify", params={"thread_id": "v2", "command": "exit 1"})
                assert r.status_code == 200
                data = r.json()
                assert data["passed"] is False
                assert data["returncode"] == 1
        finally:
            clear_active_shadow(1, "v2")
            app.dependency_overrides.clear()

    def test_verify_no_shadow(self, tmp_path, monkeypatch):
        _setup_workspace(tmp_path, monkeypatch)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/verify", params={"thread_id": "v3", "command": "echo ok"})
                assert r.status_code == 400
        finally:
            app.dependency_overrides.clear()


class TestSyncWithVerify:
    def test_sync_verify_pass(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "sv1")
        ShadowManager.create_shadow(real_ws, shadow_path)
        (Path(shadow_path) / "file1.py").write_text("modified", encoding="utf-8")
        set_active_shadow(1, "sv1", shadow_path)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/sync", params={"thread_id": "sv1", "verify_command": "echo ok"})
                assert r.status_code == 200
                data = r.json()
                assert data["synced"] == 1
                assert data["verified"] is True
                assert (Path(real_ws) / "file1.py").read_text() == "modified"
        finally:
            clear_active_shadow(1, "sv1")
            app.dependency_overrides.clear()

    def test_sync_verify_fail_blocks_sync(self, tmp_path, monkeypatch):
        real_ws = _setup_workspace(tmp_path, monkeypatch)
        shadow_path = get_shadow_path(1, "sv2")
        ShadowManager.create_shadow(real_ws, shadow_path)
        (Path(shadow_path) / "file1.py").write_text("modified", encoding="utf-8")
        set_active_shadow(1, "sv2", shadow_path)
        app.dependency_overrides[get_current_user] = _fake_user
        try:
            with TestClient(app) as c:
                r = c.post("/workspace/sync", params={"thread_id": "sv2", "verify_command": "exit 1"})
                assert r.status_code == 200
                data = r.json()
                assert data["synced"] == 0
                assert data["verified"] is False
                assert (Path(real_ws) / "file1.py").read_text() == "original"
        finally:
            clear_active_shadow(1, "sv2")
            app.dependency_overrides.clear()
