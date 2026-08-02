"""Shared helpers for capturing and comparing Brane committed results."""
import re
import shutil
import threading
import uuid
from pathlib import Path

BRANE_DATA_DIR = Path.home() / ".local" / "share" / "brane" / "data"

# Serialize the read+delete phase so parallel workers can't read each other's
# uniquely-named datasets while we're in the middle of a capture.
_capture_lock = threading.Lock()


def extract_commit_names(bs_code: str) -> list[str]:
    """Return all commit_result("name", ...) names found in bs_code."""
    return re.findall(r'commit_result\s*\(\s*"([^"]+)"', bs_code)


def pre_clean_committed(names: list[str]) -> None:
    """
    Delete committed dataset directories BEFORE running a script.
    Prevents stale data from a previous (possibly crashed) run from being
    captured as if it were produced by the current run.
    """
    for name in names:
        shutil.rmtree(BRANE_DATA_DIR / name, ignore_errors=True)


def patch_commit_names(bs_code: str) -> tuple[str, dict[str, str]]:
    """
    Rewrite every commit_result("name", ...) to commit_result("name__<uid>", ...)
    so parallel workers never write to the same Brane dataset directory.

    Returns:
        patched_code  – BraneScript with unique commit names
        name_map      – {original_name: unique_name}
    """
    uid = uuid.uuid4().hex[:8]
    name_map: dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        orig = m.group(1)
        unique = f"{orig}__{uid}"
        name_map[orig] = unique
        return f'commit_result("{unique}"'

    patched = re.sub(r'commit_result\s*\(\s*"([^"]+)"', _replace, bs_code)
    return patched, name_map


def read_and_clear_committed(
    names: list[str],
    name_map: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """
    While holding _capture_lock, read every file in each committed dataset
    directory then immediately remove the directory.

    Args:
        names     – original commit names (as they appear in the BraneScript
                    before any patching)
        name_map  – if patch_commit_names() was used, pass its name_map here
                    so we read the uniquely-named directories instead
                    (results are keyed by original names)

    Returns {original_name: {filename: text_content}}.
    """
    if not names:
        return {}
    results = {}
    with _capture_lock:
        for orig in names:
            actual = name_map[orig] if (name_map and orig in name_map) else orig
            data_dir = BRANE_DATA_DIR / actual / "data"
            if not data_dir.exists():
                continue
            files: dict[str, str] = {}
            for f in sorted(data_dir.iterdir()):
                if f.is_file():
                    try:
                        files[f.name] = f.read_text(encoding="utf-8", errors="replace").strip()
                    except Exception:
                        files[f.name] = f"<unreadable {f.stat().st_size}B>"
            if files:
                results[orig] = files  # always keyed by original name
            shutil.rmtree(BRANE_DATA_DIR / actual, ignore_errors=True)
    return results


def compare_committed(ref: dict, model: dict) -> bool | None:
    """
    Compare reference and model committed results.

    Returns:
        True   – same content (row-order-normalised)
        False  – different content
        None   – no reference committed results to compare against
    """
    if not ref:
        return None

    def _normalise(committed: dict) -> list[str]:
        rows: list[str] = []
        for files in committed.values():
            if isinstance(files, dict):
                for content in files.values():
                    rows.extend(
                        line.strip()
                        for line in content.splitlines()
                        if line.strip()
                    )
        return sorted(rows)

    return _normalise(ref) == _normalise(model)
