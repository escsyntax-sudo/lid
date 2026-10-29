from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

DEFAULT_EXTENSIONS = (".md", ".markdown", ".mdx", ".txt", ".rst", ".pdf")
DEFAULT_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    "target",
    "vendor",
)


def default_db_path() -> Path:
    override = os.environ.get("LID_DB")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "lid" / "index.db"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "lid" / "index.db"


def resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def should_skip_dir(name: str, excludes: Sequence[str]) -> bool:
    return name in excludes


def iter_files(
    paths: Iterable[Path | str],
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> Iterator[Path]:
    seen: set[str] = set()
    normalized = tuple(ext.lower() if ext.startswith(".") else "." + ext.lower() for ext in extensions)
    for raw in paths:
        path = resolve(raw)
        if path.is_file():
            if path.suffix.lower() in normalized:
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    yield path
            continue
        if not path.is_dir():
            continue
        for root, dirs, files in os.walk(path):
            dirs[:] = [name for name in dirs if not should_skip_dir(name, excludes)]
            for name in files:
                candidate = Path(root) / name
                if candidate.suffix.lower() not in normalized:
                    continue
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                yield candidate


def display_path(path: str, cwd: Optional[Path] = None) -> str:
    cwd = cwd or Path.cwd()
    try:
        return str(Path(path).resolve().relative_to(cwd))
    except (ValueError, OSError):
        return path


def human_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)
