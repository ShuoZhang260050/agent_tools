import os
import shutil
import fnmatch
import hashlib
import subprocess
import tempfile
import threading
from pathlib import Path

SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "__pycache__",
    ".pytest_cache", "dist", "build", ".superpowers",
})
MAX_SHADOW_BYTES = 200 * 1024 * 1024


def _should_skip(name: str) -> bool:
    return name in SKIP_DIRS


def _load_gitignore(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    patterns = []
    try:
        for line in gi.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    except Exception:
        pass
    return patterns


def _matches_gitignore(rel_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    rel = rel_path.replace("\\", "/")
    name = Path(rel).name
    for pat in patterns:
        pat_dir = pat.rstrip("/")
        if pat_dir and (rel == pat_dir or rel.startswith(pat_dir + "/")):
            return True
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, "/" + pat):
            return True
        if "/" not in pat and fnmatch.fnmatch(name, pat):
            return True
    return False


def _walk_files(root_path: str) -> dict[str, str]:
    result = {}
    root = Path(root_path)
    if not root.is_dir():
        return result
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not _should_skip(d)]
        rel_root = Path(dirpath).relative_to(root)
        for f in files:
            if f.startswith("."):
                continue
            rel = str(rel_root / f) if str(rel_root) != "." else f
            result[rel] = dirpath
    return result


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        pass
    return h.hexdigest()


class ShadowManager:
    @staticmethod
    def create_shadow(real_path: str, shadow_path: str) -> dict:
        src = Path(real_path).resolve()
        dst = Path(shadow_path)
        if not src.is_dir():
            raise ValueError(f"工作空间路径不是目录: {src}")
        dst.mkdir(parents=True, exist_ok=True)

        gitignore_patterns = _load_gitignore(src)
        total_bytes = 0
        file_count = 0
        skipped = []

        for dirpath, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if not _should_skip(d)]
            rel_root = Path(dirpath).relative_to(src)
            for fname in files:
                if fname.startswith("."):
                    continue
                rel = str(rel_root / fname) if str(rel_root) != "." else fname
                if _matches_gitignore(rel, gitignore_patterns):
                    skipped.append(rel)
                    continue

                src_file = Path(dirpath) / fname
                try:
                    fsize = src_file.stat().st_size
                except OSError:
                    continue
                if total_bytes + fsize > MAX_SHADOW_BYTES:
                    raise ValueError(
                        f"Shadow 大小超过上限 {MAX_SHADOW_BYTES // (1024 * 1024)}MB "
                        f"(已拷贝 {total_bytes // (1024 * 1024)}MB)"
                    )

                dst_file = dst / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src_file, dst_file)
                except OSError:
                    skipped.append(rel)
                    continue
                total_bytes += fsize
                file_count += 1

        return {"files": file_count, "bytes": total_bytes, "skipped": skipped}

    @staticmethod
    def list_shadow_diff(shadow_path: str, real_path: str) -> dict:
        shadow = Path(shadow_path)
        real = Path(real_path)
        gitignore_patterns = _load_gitignore(real)

        shadow_files = {}
        for dirpath, dirs, files in os.walk(shadow):
            dirs[:] = [d for d in dirs if not _should_skip(d)]
            rel_root = Path(dirpath).relative_to(shadow)
            for f in files:
                if f.startswith("."):
                    continue
                rel = str(rel_root / f) if str(rel_root) != "." else f
                shadow_files[rel] = os.path.join(dirpath, f)

        real_files = {}
        for dirpath, dirs, files in os.walk(real):
            dirs[:] = [d for d in dirs if not _should_skip(d)]
            rel_root = Path(dirpath).relative_to(real)
            for f in files:
                if f.startswith("."):
                    continue
                rel = str(rel_root / f) if str(rel_root) != "." else f
                if _matches_gitignore(rel, gitignore_patterns):
                    continue
                real_files[rel] = os.path.join(dirpath, f)

        added = sorted(set(shadow_files) - set(real_files))
        deleted = sorted(set(real_files) - set(shadow_files))
        modified = []
        for rel in sorted(set(shadow_files) & set(real_files)):
            s_hash = _file_hash(Path(shadow_files[rel]))
            r_hash = _file_hash(Path(real_files[rel]))
            if s_hash != r_hash:
                modified.append(rel)

        return {
            "added": added,
            "modified": modified,
            "deleted": deleted,
        }

    @staticmethod
    def apply_shadow_to_real(shadow_path: str, real_path: str) -> dict:
        diff = ShadowManager.list_shadow_diff(shadow_path, real_path)
        real = Path(real_path)
        shadow = Path(shadow_path)
        synced = 0
        total_bytes = 0

        for rel in diff["added"] + diff["modified"]:
            src = shadow / rel
            dst = real / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            synced += 1
            try:
                total_bytes += src.stat().st_size
            except OSError:
                pass

        for rel in diff["deleted"]:
            dst = real / rel
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            synced += 1

        return {"synced": synced, "bytes": total_bytes}


_active_shadows: dict[tuple[int, str], str] = {}
_shadow_lock = threading.Lock()


def get_shadow_path(user_id: int, thread_id: str) -> str:
    return os.path.join(
        tempfile.gettempdir(),
        f"agent_shadow_{user_id}_{thread_id}",
    )


def set_active_shadow(user_id: int, thread_id: str, shadow_path: str) -> None:
    with _shadow_lock:
        _active_shadows[(user_id, thread_id)] = shadow_path


def get_active_workspace(user_id: int, thread_id: str | None = None) -> str | None:
    if thread_id is None:
        return None
    with _shadow_lock:
        return _active_shadows.get((user_id, thread_id))


def clear_active_shadow(user_id: int, thread_id: str) -> None:
    with _shadow_lock:
        path = _active_shadows.pop((user_id, thread_id), None)
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def create_shadow_if_needed(user_id: int, thread_id: str) -> str | None:
    existing = get_active_workspace(user_id, thread_id)
    if existing:
        return existing
    from agent.memory.user_memory import get_workspace
    real_ws = get_workspace(user_id)
    if not real_ws:
        return None
    shadow_path = get_shadow_path(user_id, thread_id)
    ShadowManager.create_shadow(real_ws, shadow_path)
    set_active_shadow(user_id, thread_id, shadow_path)
    return shadow_path


def verify_shadow(shadow_path: str, command: str, timeout: int = 120) -> dict:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=shadow_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip()
        if len(output) > 20000:
            output = output[:20000] + "\n[...已截断]"
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "returncode": -1,
            "output": f"验证命令超时（{timeout}s）",
        }
    except Exception as e:
        return {
            "passed": False,
            "returncode": -1,
            "output": f"执行失败：{type(e).__name__}: {e}",
        }
