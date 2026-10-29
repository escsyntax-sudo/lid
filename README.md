# LID

Local Index and Documentation. A command line tool that indexes Markdown, text, reStructuredText, and PDF files on your machine, builds local vector embeddings, and answers natural language queries without a network connection.

No API keys, no model downloads, no background service. The index is a single SQLite file.

```
$ lid add ~/notes
Index: /home/you/.local/share/lid/index.db
Files: 142, added: 142, updated: 0, unchanged: 0, chunks: 986

$ lid query how do we rotate the deploy key
1. ~/notes/ops/secrets.md:12  score 0.412
   Secrets > Rotation
   Rotate the deploy key every ninety days. Run ops/rotate-key.sh and update the vault entry.
```

## Install

Python 3.9 or newer.

```sh
pip install -e .
```

For PDF support:

```sh
pip install -e ".[pdf]"
```

The package exposes a `lid` command and can also run as a module:

```sh
python -m lid --help
```

## Commands

| Command | Purpose |
| --- | --- |
| `lid add PATH...` | Index files or directories. Creates the database when needed. |
| `lid query "text"` | Search the index. Alias: `search`. |
| `lid list` | List indexed documents with chunk counts. |
| `lid stats` | Show database size, document count, chunk count, dimensions, and model. |
| `lid rm PATH` | Remove a file, or every document under a directory. |
| `lid rm --all --yes` | Remove everything. |
| `lid prune` | Drop entries whose source files no longer exist. |
| `lid reindex` | Rebuild vectors from stored term counts. |

Add `--json` to `query`, `list`, or `stats` for machine readable output.

## Query options

```sh
lid query deploy rollback -k 3 --min-score 0.08
lid query "release checklist" --full
lid query "api key" --json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-k`, `--limit` | `5` | Number of results. |
| `--min-score` | `0.05` | Minimum cosine similarity. |
| `--snippet-width` | `280` | Characters shown around the first match. |
| `--full` | off | Print the complete chunk. |
| `--json` | off | Emit a JSON object with the query and hits. |

Each hit includes the file path, line range or PDF page, score, heading, and a snippet. Paths are shown relative to the current directory when possible.

## Add options

```sh
lid add ~/docs ~/work/specs --ext .md --ext .pdf --exclude archive --chunk-size 1200
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--ext` | `.md .markdown .mdx .txt .rst .pdf` | Repeatable extension filter. |
| `--exclude` | common build and VCS folders | Repeatable directory name to skip. |
| `--chunk-size` | `900` | Maximum characters per chunk. |
| `--force` | off | Reindex even when size and modification time match. |

Unchanged files are skipped. Changed files replace their old chunks. Files that cannot be parsed are reported and the rest of the batch continues.

## Database location

The default database is:

| Platform | Path |
| --- | --- |
| Linux and macOS | `~/.local/share/lid/index.db` |
| Windows | `%LOCALAPPDATA%\lid\index.db` |

Override it with `--db`, or set `LID_DB`:

```sh
lid --db ./project-index.db add ./docs
lid query "routing table" --db ./project-index.db
```

`--db` and `--dim` are accepted before or after the subcommand.

## How the embeddings work

LID uses a local hashed TF-IDF model. Nothing leaves the machine and there is no model file to download.

1. Text is lowercased and split into word tokens. English stopwords are dropped.
2. Word bigrams are added next to the unigrams, so phrases like `api key` carry more weight than either word alone.
3. Heading tokens are counted twice, which keeps section topics visible in short chunks.
4. Each term is hashed into a fixed 512 dimension vector with a signed hash. Sublinear term frequency is multiplied by inverse document frequency.
5. Vectors are L2 normalized and stored as little endian float32 in SQLite.

Queries go through the same tokenizer and weighting. Ranking is cosine similarity. A query term that never appears in the index contributes nothing, so unrelated queries return no hits instead of hash noise.

Inverse document frequency depends on the whole corpus. Adding or removing documents therefore rebuilds the stored vectors from the saved term counts. This keeps the index consistent and is fast for personal documentation sets. `lid reindex` forces the same pass.

### Chunking

Markdown is split at headings. Each chunk keeps its heading path (`Deploy > Rollback`), source line range, and body text. Paragraphs are packed up to the chunk size. Oversized paragraphs split on sentence boundaries, then on a sliding window with overlap. Fenced code blocks are not treated as headings, and YAML front matter is skipped.

Plain text and reStructuredText use paragraph chunks. PDFs are read page by page with `pypdf`; chunks carry a page number and a `Page N` heading.

## Storage schema

SQLite in WAL mode with foreign keys:

- `docs`: absolute path, title, mtime, size, SHA-256, indexed timestamp.
- `chunks`: document id, sequence, heading, text, line range, page, term counts as JSON, vector BLOB, dimension.
- `meta`: schema version, embedding name, dimension, creation time.

Vectors are checked against the stored dimension when the database opens. Opening with a different `--dim` fails with a clear message, rather than mixing incompatible vectors.

## Python API

```python
from lid import Index

with Index("notes.db") as index:
    index.add_paths(["~/notes"], max_chars=900)
    hits = index.search("rotate the deploy key", limit=3, min_score=0.08)
    for hit in hits:
        print(hit.score, hit.path, hit.start_line, hit.heading)
```

Useful methods:

| Method | Result |
| --- | --- |
| `add_paths(paths, extensions=..., excludes=..., max_chars=900, force=False)` | Summary dict. |
| `search(query, limit=5, min_score=0.0)` | List of `SearchHit`. |
| `documents()` | Document metadata with chunk counts. |
| `stats()` | Counts, size, model, and dimension. |
| `remove(path)` | Number of documents removed. |
| `remove_all()` | Number of documents removed. |
| `prune()` | Number of missing documents removed. |
| `rebuild_vectors()` | Number of chunks re-embedded. |

`SearchHit` has `score`, `path`, `title`, `heading`, `text`, `start_line`, `end_line`, `page`, and `to_dict()`.

## Design notes

- The index is a single file, so it is easy to back up, move, or delete.
- Hashing uses `blake2b` rather than Python's `hash`, which is randomized per process. Vectors stay identical across runs.
- Document frequency is stored implicitly through term counts. There is no separate vocabulary table to keep in sync.
- Parsing is separate from ranking. `lid.parsers` can be used on its own.
- The default model favors exact and near exact vocabulary. It does not know synonyms. For personal specs and notes this is predictable and fast; it is not a replacement for a neural embedding model.
- No telemetry, no network calls, and no hidden cache directory outside the one database path.

## Limitations

- Scanned PDFs without a text layer produce no chunks. OCR is not included.
- Very large collections rebuild vectors on each add. Tens of thousands of chunks are still fine, but this is not designed for a shared search cluster.
- English stopwords are built in. Other languages are indexed, but their common words are not filtered.
- Code blocks are indexed as text. Symbol aware search is out of scope.

## Tests

```sh
python -m unittest discover -s tests
```

The suite covers Markdown chunking, front matter and code fences, BOM handling, PDF dependency errors, tokenization, hashing determinism, vector normalization, ranking, incremental updates, removal, pruning, dimension checks, JSON output, and CLI exit codes.

## License

MIT. See `LICENSE`.
