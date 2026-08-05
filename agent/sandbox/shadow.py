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
VERIFY_TIMEOUT = 120


def _get_skip_dirs() -> frozenset[str]:
    """从 Settings 读取 shadow 跳过目录集合，失败回退默认值。"""
    try:
        from agent.config import Settings
        s = Settings()
        parts = [d.strip() for d in s.shadow_skip_dirs.split(",") if d.strip()]
        return frozenset(parts) if parts else SKIP_DIRS
    except Exception:
        return SKIP_DIRS


def _get_max_bytes() -> int:
    """从 Settings 读取 shadow 大小上限，失败回退默认值。"""
    try:
        from agent.config import Settings
        return Settings().shadow_max_bytes
    except Exception:
        return MAX_SHADOW_BYTES


def _get_verify_timeout() -> int:
    """从 Settings 读取验证命令超时秒数，失败回退默认值。"""
    try:
        from agent.config import Settings
        return Settings().shadow_verify_timeout
    except Exception:
        return VERIFY_TIMEOUT


def _load_gitignore(root: Path) -> list[str]:
    """加载 .gitignore 文件中的 glob 模式列表。"""
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
    """检查相对路径是否匹配 gitignore 模式。"""
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


def _file_hash(path: Path) -> str:
    """计算文件内容的 MD5 哈希值。"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        pass
    return h.hexdigest()


class ShadowManager:
    """Shadow 工作空间管理器，提供创建/对比/同步/验证功能。"""

    @staticmethod
    def create_shadow(real_path: str, shadow_path: str) -> dict:
        """过滤拷贝真实工作空间到 shadow 路径，返回文件数和字节数。"""
        src = Path(real_path).resolve()
        dst = Path(shadow_path)
        if not src.is_dir():
            raise ValueError(f"工作空间路径不是目录: {src}")
        dst.mkdir(parents=True, exist_ok=True)

        skip_dirs = _get_skip_dirs()
        max_bytes = _get_max_bytes()
        gitignore_patterns = _load_gitignore(src)
        total_bytes = 0
        file_count = 0
        skipped = []

        for dirpath, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
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
                if total_bytes + fsize > max_bytes:
                    raise ValueError(
                        f"Shadow 大小超过上限 {max_bytes // (1024 * 1024)}MB "
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
        """对比 shadow 与真实工作空间的差异，返回 added/modified/deleted。"""
        shadow = Path(shadow_path)
        real = Path(real_path)
        skip_dirs = _get_skip_dirs()
        gitignore_patterns = _load_gitignore(real)

        shadow_files = {}
        for dirpath, dirs, files in os.walk(shadow):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            rel_root = Path(dirpath).relative_to(shadow)
            for f in files:
                if f.startswith("."):
                    continue
                rel = str(rel_root / f) if str(rel_root) != "." else f
                shadow_files[rel] = os.path.join(dirpath, f)

        real_files = {}
        for dirpath, dirs, files in os.walk(real):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
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
        """将 shadow 变更同步到真实工作空间。"""
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
    """计算指定用户和会话的 shadow 临时路径。"""
    return os.path.join(
        tempfile.gettempdir(),
        f"agent_shadow_{user_id}_{thread_id}",
    )


def set_active_shadow(user_id: int, thread_id: str, shadow_path: str) -> None:
    """设置当前活跃的 shadow 路径。"""
    with _shadow_lock:
        _active_shadows[(user_id, thread_id)] = shadow_path


def get_active_workspace(user_id: int, thread_id: str | None = None) -> str | None:
    """获取当前活跃的 shadow 路径，无则返回 None。"""
    if thread_id is None:
        return None
    with _shadow_lock:
        return _active_shadows.get((user_id, thread_id))


def clear_active_shadow(user_id: int, thread_id: str) -> None:
    """清除活跃 shadow 并删除临时目录。"""
    with _shadow_lock:
        path = _active_shadows.pop((user_id, thread_id), None)
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def clear_all_user_shadows(user_id: int) -> None:
    """清除指定用户的所有活跃 shadow（工作空间变更时调用）。"""
    with _shadow_lock:
        keys = [k for k in _active_shadows if k[0] == user_id]
        paths = [_active_shadows.pop(k) for k in keys]
    for path in paths:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def create_shadow_if_needed(user_id: int, thread_id: str) -> str | None:
    """如 shadow 不存在则创建，已存在则复用。"""
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


def verify_shadow(shadow_path: str, command: str, timeout: int = None) -> dict:
    """在 shadow 中运行验证命令，返回通过状态和输出。"""
    if timeout is None:
        timeout = _get_verify_timeout()
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
