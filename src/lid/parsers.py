from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}
TEXT_SUFFIXES = {".txt", ".rst"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
BLANK_RE = re.compile(r"^\s*$")


class ParserError(Exception):
    pass


@dataclass
class Chunk:
    text: str
    heading: str
    start_line: int
    end_line: int
    page: Optional[int] = None

    @property
    def label(self) -> str:
        if self.page:
            return f"page {self.page}"
        lines = f"{self.start_line}-{self.end_line}" if self.start_line != self.end_line else str(self.start_line)
        return f"line {lines}"


@dataclass
class ParsedDocument:
    title: str
    chunks: List[Chunk]


def _split_long(text: str, max_chars: int, overlap: int = 120) -> List[str]:
    sentences = [part for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        sentences = [text]
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            start = 0
            step = max(1, max_chars - overlap)
            while start < len(sentence):
                pieces.append(sentence[start : start + max_chars])
                start += step
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = sentence if not current else current + " " + sentence
    if current:
        pieces.append(current)
    return [piece.strip() for piece in pieces if piece.strip()]


def _chunk_blocks(
    blocks: Sequence[Tuple[int, int, str]],
    heading: str,
    max_chars: int,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_parts: List[str] = []
    current_start = 0
    current_end = 0
    current_length = 0

    def flush() -> None:
        nonlocal current_parts, current_length
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        if text:
            chunks.append(Chunk(text=text, heading=heading, start_line=current_start, end_line=current_end))
        current_parts = []
        current_length = 0

    for start, end, text in blocks:
        text = text.strip()
        if not text:
            continue
        if len(text) > max_chars:
            flush()
            for piece in _split_long(text, max_chars):
                chunks.append(Chunk(text=piece, heading=heading, start_line=start, end_line=end))
            continue
        extra = len(text) + (2 if current_parts else 0)
        if current_parts and current_length + extra > max_chars:
            flush()
        if not current_parts:
            current_start = start
        current_parts.append(text)
        current_end = end
        current_length += extra

    flush()
    return chunks


def _paragraph_blocks(lines: Sequence[Tuple[int, str]]) -> List[Tuple[int, int, str]]:
    blocks: List[Tuple[int, int, str]] = []
    buffer: List[str] = []
    start = 0
    end = 0
    for number, text in lines:
        if BLANK_RE.match(text):
            if buffer:
                blocks.append((start, end, "\n".join(buffer)))
                buffer = []
            continue
        if not buffer:
            start = number
        buffer.append(text.rstrip())
        end = number
    if buffer:
        blocks.append((start, end, "\n".join(buffer)))
    return blocks


def parse_markdown(text: str, max_chars: int = 900) -> ParsedDocument:
    lines = text.splitlines()
    start_index = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                start_index = index + 1
                break

    title = ""
    stack: List[str] = []
    sections: List[Tuple[str, int, List[Tuple[int, str]]]] = []
    heading = ""
    section_start = start_index + 1
    section_lines: List[Tuple[int, str]] = []
    in_fence = False

    def flush_section() -> None:
        nonlocal section_lines
        if section_lines:
            sections.append((heading, section_start, section_lines))
        section_lines = []

    for index in range(start_index, len(lines)):
        number = index + 1
        raw = lines[index]
        if FENCE_RE.match(raw):
            in_fence = not in_fence
            section_lines.append((number, raw))
            continue
        match = HEADING_RE.match(raw) if not in_fence else None
        if match:
            flush_section()
            level = len(match.group(1))
            label = match.group(2).strip()
            if level == 1 and not title:
                title = label
            stack = stack[: level - 1] + [label]
            heading = " > ".join(stack)
            section_start = number
        else:
            section_lines.append((number, raw))
    flush_section()

    chunks: List[Chunk] = []
    for section_heading, line_number, section in sections:
        blocks = _paragraph_blocks(section)
        for chunk in _chunk_blocks(blocks, section_heading, max_chars):
            if chunk.start_line == 0:
                chunk.start_line = line_number
                chunk.end_line = line_number
            chunks.append(chunk)

    if not title:
        title = ""
    return ParsedDocument(title=title, chunks=chunks)


def parse_plain(text: str, max_chars: int = 900, page: Optional[int] = None) -> List[Chunk]:
    numbered = [(index + 1, line) for index, line in enumerate(text.splitlines())]
    blocks = _paragraph_blocks(numbered)
    chunks = _chunk_blocks(blocks, "", max_chars)
    if page is not None:
        for chunk in chunks:
            chunk.page = page
    return chunks


def parse_pdf(path: Path, max_chars: int = 900) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParserError(
            "PDF support needs pypdf. Install it with: pip install 'lid-docs[pdf]'"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ParserError(f"could not open PDF: {exc}") from exc

    chunks: List[Chunk] = []
    metadata = reader.metadata
    title = ""
    if metadata and metadata.title:
        title = str(metadata.title).strip()
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise ParserError(f"could not read page {page_number}: {exc}") from exc
        if not text.strip():
            continue
        page_chunks = parse_plain(text, max_chars=max_chars, page=page_number)
        for chunk in page_chunks:
            if not chunk.heading:
                chunk.heading = f"Page {page_number}"
            chunks.append(chunk)
    if not title:
        title = path.stem
    return ParsedDocument(title=title, chunks=chunks)


def parse_file(path: Path, max_chars: int = 900) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, max_chars=max_chars)
    if suffix in MARKDOWN_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_markdown(text, max_chars=max_chars)
        if not parsed.title:
            parsed.title = path.stem
        return parsed
    if suffix in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(title=path.stem, chunks=parse_plain(text, max_chars=max_chars))
    raise ParserError(f"unsupported file type: {suffix or 'no extension'}")
