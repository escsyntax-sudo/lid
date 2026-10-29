from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lid.cli import main  # noqa: E402


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lid-cli-"))
        self.db = self.tmp / "index.db"
        self.doc = self.tmp / "guide.md"
        self.doc.write_text(
            "# Guide\n\nDeploy the service with the release script.\n", encoding="utf-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *args: str):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--db", str(self.db), *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_add_query_and_stats(self) -> None:
        code, out, err = self.run_cli("add", str(self.doc))
        self.assertEqual(code, 0, err)
        self.assertIn("added: 1", out)

        code, out, err = self.run_cli("query", "release", "script")
        self.assertEqual(code, 0, err)
        self.assertIn("guide.md", out)
        self.assertIn("Deploy the service", out)

        code, out, _ = self.run_cli("stats", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["documents"], 1)
        self.assertEqual(payload["chunks"], 1)

    def test_query_json(self) -> None:
        self.run_cli("add", str(self.doc))
        code, out, err = self.run_cli("query", "deploy", "--json")
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(len(payload["hits"]), 1)
        self.assertEqual(payload["hits"][0]["title"], "Guide")

    def test_list_and_remove(self) -> None:
        self.run_cli("add", str(self.doc))
        code, out, _ = self.run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn("guide.md", out)

        code, out, _ = self.run_cli("rm", str(self.doc))
        self.assertEqual(code, 0)
        self.assertIn("Removed 1 document", out)

        code, out, _ = self.run_cli("list")
        self.assertIn("No documents indexed", out)

    def test_remove_all_requires_confirmation(self) -> None:
        self.run_cli("add", str(self.doc))
        code, _, err = self.run_cli("rm", "--all")
        self.assertEqual(code, 2)
        self.assertIn("--yes", err)
        code, out, _ = self.run_cli("rm", "--all", "--yes")
        self.assertEqual(code, 0)
        self.assertIn("Removed 1 document", out)

    def test_missing_file_fails_cleanly(self) -> None:
        code, out, _ = self.run_cli("query", "nothing")
        self.assertEqual(code, 0)
        self.assertIn("No matches", out)

    def test_bad_dimension_exits_cleanly(self) -> None:
        self.run_cli("add", str(self.doc), "--dim", "64")
        code, _, err = self.run_cli("stats", "--dim", "128")
        self.assertEqual(code, 2)
        self.assertIn("dimensions", err)


if __name__ == "__main__":
    unittest.main()
