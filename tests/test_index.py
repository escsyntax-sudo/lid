from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lid.embed import DEFAULT_DIM  # noqa: E402
from lid.index import Index, IndexErrorLid, QueryError  # noqa: E402


class IndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lid-test-"))
        self.db = self.tmp / "index.db"
        self.docs = self.tmp / "docs"
        self.docs.mkdir()
        (self.docs / "deploy.md").write_text(
            "# Deployment\n\nDeploy the API with Docker Compose.\n\nRun docker compose up on the server.\n",
            encoding="utf-8",
        )
        (self.docs / "cooking.md").write_text(
            "# Bread\n\nBake bread at 220 degrees for thirty minutes.\n",
            encoding="utf-8",
        )
        (self.docs / "keys.md").write_text(
            "# API keys\n\nThe API key lives in the configuration file. Rotate the key every ninety days.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_and_search(self) -> None:
        with Index(self.db) as index:
            summary = index.add_paths([self.docs])
            self.assertEqual(summary["added"], 3)
            self.assertEqual(summary["failed"], [])
            hits = index.search("docker compose deployment", limit=3)
            self.assertTrue(hits)
            self.assertEqual(Path(hits[0].path).name, "deploy.md")
            self.assertGreater(hits[0].score, 0.1)
            self.assertIn("Docker", hits[0].text)

    def test_unchanged_files_are_skipped(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            summary = index.add_paths([self.docs])
            self.assertEqual(summary["skipped"], 3)
            self.assertEqual(summary["added"], 0)
            self.assertEqual(summary["updated"], 0)

    def test_update_replaces_chunks(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            target = self.docs / "cooking.md"
            target.write_text("# Sourdough\n\nFeed the starter with rye flour.\n", encoding="utf-8")
            summary = index.add_paths([self.docs], force=True)
            self.assertGreaterEqual(summary["updated"], 1)
            hits = index.search("rye starter", limit=2)
            self.assertEqual(Path(hits[0].path).name, "cooking.md")
            with self.assertRaises(QueryError):
                index.search("220 degrees bread", limit=3)

    def test_remove_and_prune(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            removed = index.remove(self.docs / "keys.md")
            self.assertEqual(removed, 1)
            names = [Path(doc["path"]).name for doc in index.documents()]
            self.assertNotIn("keys.md", names)

            (self.docs / "cooking.md").unlink()
            pruned = index.prune()
            self.assertEqual(pruned, 1)
            self.assertEqual(len(index.documents()), 1)

    def test_remove_all(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            self.assertEqual(index.remove_all(), 3)
            self.assertEqual(index.documents(), [])
            self.assertEqual(index.search("docker"), [])

    def test_stats_and_reindex(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            stats = index.stats()
            self.assertEqual(stats["documents"], 3)
            self.assertGreater(stats["chunks"], 0)
            self.assertEqual(stats["dimensions"], DEFAULT_DIM)
            count = index.rebuild_vectors()
            self.assertEqual(count, stats["chunks"])

    def test_dimension_mismatch_is_rejected(self) -> None:
        with Index(self.db, dim=64) as index:
            index.add_paths([self.docs])
        with self.assertRaises(IndexErrorLid):
            Index(self.db, dim=128)

    def test_empty_query_is_rejected(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            with self.assertRaises(QueryError):
                index.search("the and of")

    def test_min_score_filters(self) -> None:
        with Index(self.db) as index:
            index.add_paths([self.docs])
            hits = index.search("docker", min_score=0.99)
            self.assertEqual(hits, [])

    def test_unsupported_files_are_reported(self) -> None:
        odd = self.docs / "notes.bin"
        odd.write_bytes(b"\x00\x01")
        with Index(self.db) as index:
            summary = index.add_paths([odd], extensions=[".md", ".bin"])
            self.assertEqual(len(summary["failed"]), 1)
            self.assertIn("unsupported", summary["failed"][0][1])


if __name__ == "__main__":
    unittest.main()
