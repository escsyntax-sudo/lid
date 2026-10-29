from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lid.parsers import ParsedDocument, parse_markdown, parse_plain  # noqa: E402


class MarkdownTests(unittest.TestCase):
    def test_headings_and_chunks(self) -> None:
        text = (
            "# Guide\n\n"
            "Intro paragraph.\n\n"
            "## Install\n\n"
            "Run the installer.\n\n"
            "## Usage\n\n"
            "Call the command.\n"
        )
        parsed = parse_markdown(text, max_chars=200)
        self.assertIsInstance(parsed, ParsedDocument)
        self.assertEqual(parsed.title, "Guide")
        headings = [chunk.heading for chunk in parsed.chunks]
        self.assertEqual(headings, ["Guide", "Guide > Install", "Guide > Usage"])
        self.assertEqual(parsed.chunks[1].text, "Run the installer.")
        self.assertEqual(parsed.chunks[1].start_line, 7)

    def test_frontmatter_is_skipped(self) -> None:
        text = "---\ntitle: Hidden\n---\n# Real\n\nBody text.\n"
        parsed = parse_markdown(text)
        self.assertEqual(parsed.title, "Real")
        self.assertEqual(parsed.chunks[0].text, "Body text.")

    def test_fenced_code_heading_is_not_a_heading(self) -> None:
        text = "# Title\n\n```sh\n# not a heading\necho hi\n```\n"
        parsed = parse_markdown(text)
        self.assertEqual(len(parsed.chunks), 1)
        self.assertIn("# not a heading", parsed.chunks[0].text)

    def test_long_paragraph_splits(self) -> None:
        sentence = "This sentence repeats to make a long paragraph."
        text = "# Long\n\n" + " ".join([sentence] * 40)
        parsed = parse_markdown(text, max_chars=200)
        self.assertGreater(len(parsed.chunks), 3)
        for chunk in parsed.chunks:
            self.assertLessEqual(len(chunk.text), 220)
            self.assertTrue(chunk.text)

    def test_plain_text_chunks(self) -> None:
        chunks = parse_plain("First.\n\nSecond paragraph.\n", max_chars=100)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Second paragraph", chunks[0].text)

    def test_bom_is_stripped_from_markdown(self) -> None:
        import tempfile

        from lid.parsers import parse_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.md"
            path.write_text("\ufeff# Title\n\nBody.\n", encoding="utf-8")
            parsed = parse_file(path)
            self.assertEqual(parsed.title, "Title")
            self.assertNotIn("\ufeff", parsed.chunks[0].text)

    def test_pdf_without_pypdf_reports_install_hint(self) -> None:
        from unittest.mock import patch

        from lid.parsers import ParserError, parse_pdf

        with patch.dict(sys.modules, {"pypdf": None}):
            with self.assertRaises(ParserError) as caught:
                parse_pdf(Path("missing.pdf"))
        self.assertIn("pypdf", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
