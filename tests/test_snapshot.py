import json
import os
from pathlib import Path

import pytest

from agent.sandbox.snapshot import (
    init_snapshots_table,
    save_snapshot,
    restore_snapshot,
    list_snapshots,
    get_latest_snapshot_id,
)


@pytest.fixture
def db_path(tmp_path):
    db = str(tmp_path / "test_snapshots.sqlite")
    init_snapshots_table(db)
    return db


@pytest.fixture
def real_ws(tmp_path):
    ws = tmp_path / "real_ws"
    ws.mkdir()
    (ws / "file1.py").write_text("original", encoding="utf-8")
    (ws / "file2.py").write_text("original2", encoding="utf-8")
    return str(ws)


class TestSaveSnapshot:
    def test_save_creates_record(self, db_path, real_ws):
        diff = {"added": ["new.py"], "modified": ["file1.py"], "deleted": []}
        sid = save_snapshot(db_path, 1, "t1", real_ws, diff)
        assert sid > 0

    def test_save_copies_files(self, db_path, real_ws):
        diff = {"added": [], "modified": ["file1.py"], "deleted": []}
        sid = save_snapshot(db_path, 1, "t1", real_ws, diff)
        snaps = list_snapshots(db_path, 1, "t1")
        assert len(snaps) == 1
        snap_path = snaps[0]["diff"]
        assert snaps[0]["diff"]["modified"] == ["file1.py"]


class TestListSnapshots:
    def test_list_empty(self, db_path):
        assert list_snapshots(db_path, 1, "t1") == []

    def test_list_multiple(self, db_path, real_ws):
        diff = {"added": [], "modified": ["file1.py"], "deleted": []}
        save_snapshot(db_path, 1, "t1", real_ws, diff)
        save_snapshot(db_path, 1, "t1", real_ws, diff)
        snaps = list_snapshots(db_path, 1, "t1")
        assert len(snaps) == 2
        assert snaps[0]["id"] > snaps[1]["id"]

    def test_list_filtered_by_thread(self, db_path, real_ws):
        diff = {"added": [], "modified": ["file1.py"], "deleted": []}
        save_snapshot(db_path, 1, "t1", real_ws, diff)
        save_snapshot(db_path, 1, "t2", real_ws, diff)
        assert len(list_snapshots(db_path, 1, "t1")) == 1
        assert len(list_snapshots(db_path, 1, "t2")) == 1


class TestRestoreSnapshot:
    def test_restore_modified(self, db_path, real_ws):
        diff = {"added": [], "modified": ["file1.py"], "deleted": []}
        sid = save_snapshot(db_path, 1, "t1", real_ws, diff)

        (Path(real_ws) / "file1.py").write_text("changed", encoding="utf-8")

        result = restore_snapshot(db_path, sid)
        assert result["restored"] == 1
        assert (Path(real_ws) / "file1.py").read_text() == "original"

    def test_restore_deleted(self, db_path, real_ws):
        diff = {"added": [], "modified": [], "deleted": ["file2.py"]}
        sid = save_snapshot(db_path, 1, "t1", real_ws, diff)

        (Path(real_ws) / "file2.py").unlink()
        assert not (Path(real_ws) / "file2.py").exists()

        restore_snapshot(db_path, sid)
        assert (Path(real_ws) / "file2.py").exists()

    def test_restore_nonexistent_raises(self, db_path):
        with pytest.raises(ValueError, match="不存在"):
            restore_snapshot(db_path, 999)


class TestGetLatestSnapshotId:
    def test_none_when_empty(self, db_path):
        assert get_latest_snapshot_id(db_path, 1, "t1") is None

    def test_returns_latest(self, db_path, real_ws):
        diff = {"added": [], "modified": ["file1.py"], "deleted": []}
        sid1 = save_snapshot(db_path, 1, "t1", real_ws, diff)
        sid2 = save_snapshot(db_path, 1, "t1", real_ws, diff)
        assert get_latest_snapshot_id(db_path, 1, "t1") == sid2
