"""LID: local index and documentation search."""

__version__ = "0.1.0"

from .index import Index, IndexErrorLid, QueryError
from .parsers import Chunk, ParserError, parse_file

__all__ = ["Index", "IndexErrorLid", "QueryError", "Chunk", "ParserError", "parse_file", "__version__"]
