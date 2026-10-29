from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from . import __version__
from .embed import DEFAULT_DIM, tokenize
from .index import Index, IndexErrorLid, QueryError
from .util import DEFAULT_EXCLUDES, DEFAULT_EXTENSIONS, display_path, eprint, human_size, plural


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lid",
        description="Index local Markdown, text, and PDF files and query them offline.",
    )
    parser.add_argument("--db", help="Index database path. Defaults to the platform data directory or LID_DB.")
    parser.add_argument("--dim", type=int, default=None, help=f"Embedding dimensions. Default {DEFAULT_DIM}.")
    parser.add_argument("--version", action="version", version=f"lid {__version__}")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--db", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    shared.add_argument("--dim", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", parents=[shared], help="Index files or directories.")
    add.add_argument("paths", nargs="+", help="Files or directories to index.")
    add.add_argument(
        "--ext",
        action="append",
        default=None,
        metavar="EXT",
        help="Extension to include, repeatable. Defaults to Markdown, text, reStructuredText, and PDF.",
    )
    add.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="NAME",
        help="Directory name to skip, repeatable.",
    )
    add.add_argument("--chunk-size", type=int, default=900, help="Maximum characters per chunk. Default 900.")
    add.add_argument("--force", action="store_true", help="Reindex files even when size and mtime match.")

    search = commands.add_parser("query", aliases=["search"], parents=[shared], help="Query the index.")
    search.add_argument("text", nargs="+", help="Query text.")
    search.add_argument("-k", "--limit", type=int, default=5, help="Number of results. Default 5.")
    search.add_argument("--min-score", type=float, default=0.05, help="Minimum cosine score. Default 0.05.")
    search.add_argument("--json", action="store_true", help="Print JSON.")
    search.add_argument("--full", action="store_true", help="Print the full chunk instead of a snippet.")
    search.add_argument("--snippet-width", type=int, default=280, help="Snippet width in characters. Default 280.")

    listing = commands.add_parser("list", aliases=["ls"], parents=[shared], help="List indexed documents.")
    listing.add_argument("--json", action="store_true", help="Print JSON.")

    remove = commands.add_parser("rm", parents=[shared], help="Remove a file or directory from the index.")
    remove.add_argument("target", nargs="?", help="File or directory to remove.")
    remove.add_argument("--all", action="store_true", help="Remove every document.")
    remove.add_argument("--yes", action="store_true", help="Skip confirmation for --all.")

    commands.add_parser("prune", parents=[shared], help="Remove entries whose files no longer exist.")
    commands.add_parser("reindex", parents=[shared], help="Rebuild vectors from stored term counts.")

    stats = commands.add_parser("stats", parents=[shared], help="Show index statistics.")
    stats.add_argument("--json", action="store_true", help="Print JSON.")

    return parser


def _open(args: argparse.Namespace) -> Index:
    dim = args.dim if getattr(args, "dim", None) is not None else DEFAULT_DIM
    return Index(db_path=args.db, dim=dim)


def _snippet(text: str, query: str, width: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    terms = tokenize(query)
    lower = flat.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    position = min(positions) if positions else 0
    start = max(0, position - width // 3)
    end = min(len(flat), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(flat) else ""
    return prefix + flat[start:end].strip() + suffix


def _cmd_add(args: argparse.Namespace) -> int:
    extensions = args.ext if args.ext else DEFAULT_EXTENSIONS
    excludes = args.exclude if args.exclude else DEFAULT_EXCLUDES
    with _open(args) as index:
        summary = index.add_paths(
            args.paths,
            extensions=extensions,
            excludes=excludes,
            max_chars=args.chunk_size,
            force=args.force,
        )
        print(f"Index: {index.db_path}")
        print(
            f"Files: {summary['files']}, added: {summary['added']}, updated: {summary['updated']}, "
            f"unchanged: {summary['skipped']}, chunks: {summary['chunks']}"
        )
        if summary["failed"]:
            eprint(f"Could not index {plural(len(summary['failed']), 'file')}:")
            for path, reason in summary["failed"]:
                eprint(f"  {path}: {reason}")
            return 1
        if summary["files"] == 0:
            print("No matching files found.")
        return 0


def _cmd_query(args: argparse.Namespace) -> int:
    query = " ".join(args.text)
    with _open(args) as index:
        hits = index.search(query, limit=args.limit, min_score=args.min_score)
        if args.json:
            payload = {"query": query, "hits": [hit.to_dict() for hit in hits]}
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        if not hits:
            print("No matches. Try different words or a lower --min-score.")
            return 0
        for number, hit in enumerate(hits, start=1):
            location = display_path(hit.path)
            if hit.page:
                location += f":page {hit.page}"
            else:
                location += f":{hit.start_line}"
            print(f"{number}. {location}  score {hit.score:.3f}")
            if hit.title or hit.heading:
                context = hit.heading or hit.title
                print(f"   {context}")
            body = hit.text if args.full else _snippet(hit.text, query, args.snippet_width)
            for line in body.splitlines():
                print(f"   {line}")
            if number != len(hits):
                print()
        return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with _open(args) as index:
        documents = index.documents()
        if args.json:
            print(json.dumps(documents, indent=2, ensure_ascii=False))
            return 0
        if not documents:
            print("No documents indexed. Run: lid add <path>")
            return 0
        for doc in documents:
            location = display_path(doc["path"])
            title = doc["title"] or Path(doc["path"]).stem
            print(f"{location}  {title}  {doc['chunks']} chunks  {human_size(doc['size'])}")
        print(f"Total: {plural(len(documents), 'document')}")
        return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    if args.all and not args.yes:
        eprint("Refusing to remove every document without --yes.")
        return 2
    if not args.all and not args.target:
        eprint("Provide a path, or use --all.")
        return 2
    with _open(args) as index:
        if args.all:
            count = index.remove_all()
            print(f"Removed {plural(count, 'document')}.")
            return 0
        count = index.remove(args.target)
        if count == 0:
            print("No indexed documents matched that path.")
            return 0
        print(f"Removed {plural(count, 'document')}.")
        return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    with _open(args) as index:
        count = index.prune()
        print(f"Removed {plural(count, 'missing document')}.")
        return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    with _open(args) as index:
        count = index.rebuild_vectors()
        print(f"Rebuilt vectors for {plural(count, 'chunk')}.")
        return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with _open(args) as index:
        stats = index.stats()
        if args.json:
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            return 0
        print(f"Database: {stats['db']}")
        print(f"Size: {stats['db_size_human']}")
        print(f"Documents: {stats['documents']}")
        print(f"Chunks: {stats['chunks']}")
        print(f"Dimensions: {stats['dimensions']}")
        print(f"Embedding: {stats['embedding']}")
        print(f"Last indexed: {stats['last_indexed'] or 'never'}")
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            return _cmd_add(args)
        if args.command in ("query", "search"):
            return _cmd_query(args)
        if args.command in ("list", "ls"):
            return _cmd_list(args)
        if args.command == "rm":
            return _cmd_rm(args)
        if args.command == "prune":
            return _cmd_prune(args)
        if args.command == "reindex":
            return _cmd_reindex(args)
        if args.command == "stats":
            return _cmd_stats(args)
    except (IndexErrorLid, QueryError, ValueError) as exc:
        eprint(f"lid: {exc}")
        return 2
    except KeyboardInterrupt:
        eprint("lid: interrupted")
        return 130
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
