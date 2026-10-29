from __future__ import annotations

import heapq
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .embed import DEFAULT_DIM, MODEL_NAME, TfidfModel, tokenize
from .parsers import Chunk, ParserError, parse_file
from .util import DEFAULT_EXCLUDES, DEFAULT_EXTENSIONS, default_db_path, human_size, resolve, sha256_file

SCHEMA_VERSION = "1"


class IndexErrorLid(Exception):
    pass


class QueryError(Exception):
    pass


@dataclass
class SearchHit:
    score: float
    path: str
    title: str
    heading: str
    text: str
    start_line: int
    end_line: int
    page: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "path": self.path,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "page": self.page,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Index:
    def __init__(self, db_path: Optional[Path | str] = None, dim: int = DEFAULT_DIM):
        self.db_path = Path(db_path).expanduser() if db_path else default_db_path()
        self.db_path = self.db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._create_schema()
        stored_dim = self._get_meta("dim")
        if stored_dim is not None and int(stored_dim) != dim:
            raise IndexErrorLid(
                f"index uses {stored_dim} dimensions, requested {dim}. "
                f"Reopen with --dim {stored_dim}, or delete the database and index again."
            )
        stored_model = self._get_meta("embedding")
        if stored_model is not None and stored_model != MODEL_NAME:
            raise IndexErrorLid(
                f"index was built with embedding '{stored_model}', this build uses '{MODEL_NAME}'. "
                "Delete the database and index again."
            )
        self.dim = int(stored_dim) if stored_dim is not None else dim
        self.model: Optional[TfidfModel] = None
        self._set_meta("schema_version", SCHEMA_VERSION)
        self._set_meta("embedding", MODEL_NAME)
        self._set_meta("dim", str(self.dim))
        if self._get_meta("created_at") is None:
            self._set_meta("created_at", _utc_now())
        self.conn.commit()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS docs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              path TEXT NOT NULL UNIQUE,
              title TEXT NOT NULL DEFAULT '',
              mtime REAL NOT NULL,
              size INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
              seq INTEGER NOT NULL,
              heading TEXT NOT NULL DEFAULT '',
              text TEXT NOT NULL,
              start_line INTEGER NOT NULL,
              end_line INTEGER NOT NULL,
              page INTEGER,
              terms TEXT NOT NULL,
              vector BLOB,
              dim INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            """
        )

    def _get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def add_paths(
        self,
        paths: Iterable[Path | str],
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        excludes: Sequence[str] = DEFAULT_EXCLUDES,
        max_chars: int = 900,
        force: bool = False,
    ) -> Dict[str, Any]:
        from .util import iter_files

        files = list(iter_files(paths, extensions=extensions, excludes=excludes))
        summary: Dict[str, Any] = {
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "chunks": 0,
            "failed": [],
            "files": len(files),
        }
        changed = False

        for path in files:
            try:
                stat = path.stat()
                doc_path = str(path)
                existing = self.conn.execute("SELECT * FROM docs WHERE path = ?", (doc_path,)).fetchone()
                if not force and existing and float(existing["mtime"]) == stat.st_mtime and int(existing["size"]) == stat.st_size:
                    summary["skipped"] += 1
                    continue

                digest = sha256_file(path)
                if not force and existing and existing["sha256"] == digest:
                    summary["skipped"] += 1
                    continue

                parsed = parse_file(path, max_chars=max_chars)
                if not parsed.chunks:
                    summary["failed"].append((doc_path, "no extractable text"))
                    continue
                title = parsed.title.strip() or path.stem
                with self.conn:
                    if existing:
                        self.conn.execute("DELETE FROM docs WHERE id = ?", (existing["id"],))
                    cursor = self.conn.execute(
                        "INSERT INTO docs(path, title, mtime, size, sha256, indexed_at) VALUES(?, ?, ?, ?, ?, ?)",
                        (doc_path, title, stat.st_mtime, stat.st_size, digest, _utc_now()),
                    )
                    doc_id = cursor.lastrowid
                    for seq, chunk in enumerate(parsed.chunks):
                        terms = self._terms_for(chunk)
                        self.conn.execute(
                            "INSERT INTO chunks(doc_id, seq, heading, text, start_line, end_line, page, terms, vector, dim) "
                            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                            (
                                doc_id,
                                seq,
                                chunk.heading,
                                chunk.text,
                                chunk.start_line,
                                chunk.end_line,
                                chunk.page,
                                json.dumps(terms, ensure_ascii=False, separators=(",", ":")),
                                self.dim,
                            ),
                        )
                summary["chunks"] += len(parsed.chunks)
                if existing:
                    summary["updated"] += 1
                else:
                    summary["added"] += 1
                changed = True
            except ParserError as exc:
                summary["failed"].append((str(path), str(exc)))
            except OSError as exc:
                summary["failed"].append((str(path), f"file error: {exc}"))
            except Exception as exc:  # keep the batch going, report the file
                summary["failed"].append((str(path), f"unexpected error: {exc}"))

        if changed:
            self.rebuild_vectors()
        return summary

    def _terms_for(self, chunk: Chunk) -> Dict[str, int]:
        from .embed import term_counts

        return term_counts(chunk.text, chunk.heading)

    def rebuild_vectors(self) -> int:
        rows = self.conn.execute("SELECT id, terms FROM chunks ORDER BY id").fetchall()
        term_dicts = [json.loads(row["terms"]) for row in rows]
        model = TfidfModel(self.dim).fit(term_dicts)
        updates = []
        for row, terms in zip(rows, term_dicts):
            updates.append((sqlite3.Binary(model.to_bytes(model.vector(terms))), row["id"]))
        with self.conn:
            self.conn.executemany("UPDATE chunks SET vector = ? WHERE id = ?", updates)
        self.model = model
        return len(updates)

    def _ensure_model(self) -> TfidfModel:
        if self.model is not None:
            return self.model
        rows = self.conn.execute("SELECT terms FROM chunks ORDER BY id").fetchall()
        self.model = TfidfModel(self.dim).fit([json.loads(row["terms"]) for row in rows])
        return self.model

    def search(self, query: str, limit: int = 5, min_score: float = 0.0) -> List[SearchHit]:
        if limit <= 0:
            raise QueryError("limit must be positive")
        terms = tokenize(query)
        if not terms:
            raise QueryError("query contains no searchable words")
        rows = self.conn.execute(
            "SELECT c.text, c.heading, c.start_line, c.end_line, c.page, c.vector, "
            "d.path, d.title FROM chunks c JOIN docs d ON d.id = c.doc_id ORDER BY c.id"
        ).fetchall()
        if not rows:
            return []
        model = self._ensure_model()
        query_vector = model.vector_from_text(query)
        if not any(query_vector):
            raise QueryError("query has no terms present in the index")

        scored: List[Tuple[float, int, sqlite3.Row]] = []
        for index, row in enumerate(rows):
            blob = row["vector"]
            if blob is None:
                continue
            score = model.similarity(query_vector, blob)
            if score >= min_score:
                scored.append((score, index, row))

        best = heapq.nlargest(limit, scored, key=lambda item: (item[0], -item[1]))
        hits: List[SearchHit] = []
        for score, _, row in best:
            hits.append(
                SearchHit(
                    score=score,
                    path=row["path"],
                    title=row["title"],
                    heading=row["heading"],
                    text=row["text"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    page=row["page"],
                )
            )
        return hits

    def documents(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT d.id, d.path, d.title, d.size, d.indexed_at, "
            "COUNT(c.id) AS chunks, COALESCE(SUM(LENGTH(c.text)), 0) AS characters "
            "FROM docs d LEFT JOIN chunks c ON c.doc_id = d.id "
            "GROUP BY d.id ORDER BY d.path"
        ).fetchall()
        return [
            {
                "id": row["id"],
                "path": row["path"],
                "title": row["title"],
                "size": row["size"],
                "indexed_at": row["indexed_at"],
                "chunks": row["chunks"],
                "characters": row["characters"],
            }
            for row in rows
        ]

    def stats(self) -> Dict[str, Any]:
        docs = self.conn.execute("SELECT COUNT(*) AS count FROM docs").fetchone()["count"]
        chunks = self.conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        last = self.conn.execute("SELECT MAX(indexed_at) AS last FROM docs").fetchone()["last"]
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db": str(self.db_path),
            "db_size": size,
            "db_size_human": human_size(size),
            "documents": docs,
            "chunks": chunks,
            "dimensions": self.dim,
            "embedding": MODEL_NAME,
            "last_indexed": last,
        }

    def remove(self, target: Path | str) -> int:
        target_path = str(resolve(target))
        rows = self.conn.execute("SELECT id, path FROM docs").fetchall()
        ids = [
            row["id"]
            for row in rows
            if row["path"] == target_path or row["path"].startswith(target_path + os.sep)
        ]
        if not ids:
            return 0
        with self.conn:
            self.conn.executemany("DELETE FROM docs WHERE id = ?", [(doc_id,) for doc_id in ids])
        self.rebuild_vectors()
        return len(ids)

    def remove_all(self) -> int:
        count = self.conn.execute("SELECT COUNT(*) AS count FROM docs").fetchone()["count"]
        with self.conn:
            self.conn.execute("DELETE FROM docs")
        self.model = TfidfModel(self.dim).fit([])
        return count

    def prune(self) -> int:
        rows = self.conn.execute("SELECT id, path FROM docs").fetchall()
        missing = [row["id"] for row in rows if not Path(row["path"]).exists()]
        if not missing:
            return 0
        with self.conn:
            self.conn.executemany("DELETE FROM docs WHERE id = ?", [(doc_id,) for doc_id in missing])
        self.rebuild_vectors()
        return len(missing)
