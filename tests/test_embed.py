from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lid.embed import TfidfModel, cosine, term_counts, tokenize  # noqa: E402


class EmbeddingTests(unittest.TestCase):
    def test_tokenize_drops_stopwords(self) -> None:
        tokens = tokenize("How do I deploy the API server?")
        self.assertEqual(tokens, ["deploy", "api", "server"])

    def test_term_counts_adds_bigrams(self) -> None:
        terms = term_counts("deploy api server", "Deploy")
        self.assertGreaterEqual(terms["deploy"], 3)
        self.assertEqual(terms["deploy api"], 1)
        self.assertEqual(terms["api server"], 1)

    def test_vectors_are_normalised_and_deterministic(self) -> None:
        model = TfidfModel(dim=64).fit([term_counts("alpha beta"), term_counts("beta gamma")])
        first = model.vector(term_counts("alpha beta"))
        second = TfidfModel(dim=64).fit([term_counts("alpha beta"), term_counts("beta gamma")]).vector(
            term_counts("alpha beta")
        )
        self.assertEqual(len(first), 64)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0, places=5)
        self.assertEqual(first, second)

    def test_round_trip_bytes(self) -> None:
        model = TfidfModel(dim=32).fit([term_counts("hello world")])
        vector = model.vector(term_counts("hello world"))
        restored = model.from_bytes(model.to_bytes(vector))
        self.assertAlmostEqual(cosine(vector, restored), 1.0, places=5)

    def test_similarity_ranks_related_text(self) -> None:
        model = TfidfModel(dim=128).fit(
            [term_counts("docker deployment"), term_counts("bread baking"), term_counts("api keys")]
        )
        query = model.vector_from_text("docker")
        related = model.vector_from_text("docker deployment")
        unrelated = model.vector_from_text("bread baking")
        self.assertGreater(cosine(query, related), cosine(query, unrelated))

    def test_unknown_terms_do_not_crash(self) -> None:
        model = TfidfModel(dim=32).fit([term_counts("known term")])
        vector = model.vector_from_text("unknown words")
        self.assertEqual(len(vector), 32)


if __name__ == "__main__":
    unittest.main()
