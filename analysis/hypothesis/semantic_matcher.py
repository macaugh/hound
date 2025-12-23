"""Semantic similarity matching for hypothesis deduplication."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    """Semantic similarity matcher using sentence transformers."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 threshold: float = 0.85):
        """Initialize semantic matcher.

        Args:
            model_name: Sentence transformer model name
            threshold: Similarity threshold for duplicate detection (0-1)
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self._cache: dict[str, np.ndarray] = {}  # Cache embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two text strings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Get embeddings
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Compute cosine similarity
        similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]

        return float(similarity)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        if text in self._cache:
            return self._cache[text]

        embedding = self.model.encode(text, convert_to_numpy=True)
        self._cache[text] = embedding

        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entry (FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        return embedding

    def is_duplicate(self, new_text: str, existing_texts: list[str]) -> tuple[bool, str | None]:
        """Check if new text is duplicate of any existing text.

        Args:
            new_text: Text to check
            existing_texts: List of existing texts

        Returns:
            Tuple of (is_duplicate, matched_text)
        """
        if not existing_texts:
            return False, None

        # Get embedding for new text
        new_emb = self._get_embedding(new_text)

        # Compare against all existing
        for existing_text in existing_texts:
            existing_emb = self._get_embedding(existing_text)

            similarity = cosine_similarity(
                new_emb.reshape(1, -1),
                existing_emb.reshape(1, -1)
            )[0][0]

            if similarity >= self.threshold:
                return True, existing_text

        return False, None
