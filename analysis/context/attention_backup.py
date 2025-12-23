"""Attention-based context selection for agent.

Implements transformer-style attention mechanism using sentence embeddings.
Computes attention scores between investigation goal and context parts.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class AttentionScorer:
    """Compute attention scores for context selection."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def compute_attention_scores(self,
                                 query: str,
                                 contexts: list[str]) -> np.ndarray:
        """Compute attention scores using cosine similarity.

        Args:
            query: Investigation goal or current focus
            contexts: List of context parts to score

        Returns:
            Array of attention scores (0-1, higher = more relevant)
        """
        # Encode query
        query_emb = self._encode(query)

        # Encode contexts
        context_embs = np.array([self._encode(ctx) for ctx in contexts])

        # Compute cosine similarities
        scores = cosine_similarity(
            query_emb.reshape(1, -1),
            context_embs
        )[0]

        # Apply softmax for probability distribution
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention = exp_scores / exp_scores.sum()

        return attention

    def _encode(self, text: str) -> np.ndarray:
        """Encode text with caching."""
        if text not in self._cache:
            self._cache[text] = self.model.encode(text, convert_to_numpy=True)

            # Limit cache size
            if len(self._cache) > 500:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return self._cache[text]


class AttentionFilter:
    """Filter context parts based on attention scores."""

    def __init__(self,
                 scorer: AttentionScorer,
                 threshold: float = 0.1,
                 top_k: int | None = None):
        """
        Args:
            scorer: AttentionScorer instance
            threshold: Minimum attention score to include (0-1)
            top_k: If set, include only top K parts (overrides threshold)
        """
        self.scorer = scorer
        self.threshold = threshold
        self.top_k = top_k

    def filter_contexts(self,
                       query: str,
                       contexts: list[tuple[str, str]],
                       critical_labels: set[str]) -> list[tuple[str, str, float]]:
        """Filter contexts by attention scores.

        Args:
            query: Investigation goal
            contexts: List of (label, content) tuples
            critical_labels: Labels that must always be included

        Returns:
            List of (label, content, attention_score) tuples
        """
        # Extract labels and content
        labels, contents = zip(*contexts) if contexts else ([], [])

        # Compute attention scores
        scores = self.scorer.compute_attention_scores(query, list(contents))

        # Build results
        results = []
        for i, (label, content) in enumerate(contexts):
            score = float(scores[i])

            # Always include critical sections
            if label in critical_labels:
                results.append((label, content, 1.0))  # Max score
                continue

            # Apply threshold or top-k filtering
            if self.top_k is not None:
                # Will filter to top-k after sorting
                results.append((label, content, score))
            elif score >= self.threshold:
                results.append((label, content, score))

        # Sort by score (descending)
        results.sort(key=lambda x: x[2], reverse=True)

        # Apply top-k if specified
        if self.top_k is not None:
            results = results[:self.top_k]

        return results
