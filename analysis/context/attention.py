"""Attention-based context selection for agent.

Implements transformer-style attention mechanism using sentence embeddings.
Computes attention scores between investigation goal and context parts.

NOTE: This module is being developed using Test-Driven Development (TDD).
All implementations are driven by tests in tests/test_attention_*.py

DESIGN DECISION: Uses raw cosine similarity (not softmax) for attention scores.
Rationale: Absolute relevance is more useful than relative scores for threshold filtering.
Scores represent semantic similarity in [0, 1] range, interpretable across different contexts.
"""

from collections import OrderedDict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AttentionScorer:
    """Compute attention scores for context selection using semantic similarity."""

    # Cache configuration
    MAX_CACHE_SIZE = 500
    EVICTION_BATCH_SIZE = 100  # Evict multiple items at once to reduce thrashing

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize scorer with sentence transformer model.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # Use OrderedDict for proper LRU tracking
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _encode(self, text: str) -> np.ndarray:
        """Encode text with LRU caching.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if text in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(text)
            return self._cache[text]

        # Not in cache - encode it
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Add to cache
        self._cache[text] = embedding

        # Evict oldest items if cache exceeded
        if len(self._cache) > self.MAX_CACHE_SIZE:
            # Remove oldest EVICTION_BATCH_SIZE items to reduce thrashing
            for _ in range(min(self.EVICTION_BATCH_SIZE, len(self._cache) - self.MAX_CACHE_SIZE + self.EVICTION_BATCH_SIZE)):
                if len(self._cache) > self.MAX_CACHE_SIZE:
                    self._cache.popitem(last=False)  # Remove oldest (FIFO)

        return embedding

    def compute_attention_scores(self, query: str, contexts: list[str]) -> np.ndarray:
        """Compute attention scores using cosine similarity.

        Uses RAW cosine similarity (not softmax) to provide absolute relevance scores.
        This allows threshold-based filtering to work reliably across different batches.

        Args:
            query: Investigation goal or current focus
            contexts: List of context parts to score

        Returns:
            Array of attention scores in [0, 1] range (higher = more relevant)
        """
        # Encode query and contexts using cache
        query_emb = self._encode(query)
        context_embs = np.array([self._encode(ctx) for ctx in contexts])

        # Compute cosine similarities
        scores = cosine_similarity(
            query_emb.reshape(1, -1),
            context_embs
        )[0]

        # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
        normalized_scores = (scores + 1.0) / 2.0

        return normalized_scores


class AttentionFilter:
    """Filter context parts based on attention scores.

    IMPORTANT: Fixes critical bugs from code review:
    1. Critical sections use ACTUAL scores (not hardcoded 1.0)
    2. top_k returns top K regular items PLUS all critical sections
    3. Threshold filtering works reliably with absolute scores
    """

    def __init__(self,
                 scorer: AttentionScorer,
                 threshold: float = 0.1,
                 top_k: int | None = None):
        """Initialize filter with scorer and filtering parameters.

        Args:
            scorer: AttentionScorer instance for computing relevance
            threshold: Minimum attention score to include (0-1)
            top_k: If set, include top K non-critical items (critical items are additional)
        """
        self.scorer = scorer
        self.threshold = threshold
        self.top_k = top_k

    def filter_contexts(self,
                       query: str,
                       contexts: list[tuple[str, str]],
                       critical_labels: set[str]) -> list[tuple[str, str, float]]:
        """Filter contexts by attention scores.

        CRITICAL BEHAVIOR (fixes bugs from code review):
        - Critical sections use their ACTUAL attention score (not hardcoded 1.0)
        - Critical sections are ALWAYS included regardless of score
        - top_k limit applies only to non-critical items
        - If top_k=3 and 2 items are critical, returns 5 items (3 + 2)

        Args:
            query: Investigation goal or current focus
            contexts: List of (label, content) tuples
            critical_labels: Labels that must always be included

        Returns:
            List of (label, content, attention_score) tuples, sorted by score descending
        """
        # Handle empty input
        if not contexts:
            return []

        # Extract labels and content
        labels, contents = zip(*contexts)

        # Compute attention scores
        scores = self.scorer.compute_attention_scores(query, list(contents))

        # Separate critical and non-critical items
        critical_items = []
        non_critical_items = []

        for i, (label, content) in enumerate(contexts):
            score = float(scores[i])

            if label in critical_labels:
                # Critical: use ACTUAL score, always include
                critical_items.append((label, content, score))
            else:
                # Non-critical: apply filtering
                non_critical_items.append((label, content, score))

        # Filter non-critical items by threshold
        if self.top_k is None:
            # Threshold-based filtering
            filtered_non_critical = [
                item for item in non_critical_items
                if item[2] >= self.threshold
            ]
        else:
            # Top-k filtering (top k non-critical items)
            # Sort by score and take top k
            non_critical_items.sort(key=lambda x: x[2], reverse=True)
            filtered_non_critical = non_critical_items[:self.top_k]

        # Combine critical (always included) + filtered non-critical
        all_items = critical_items + filtered_non_critical

        # Sort by score descending
        all_items.sort(key=lambda x: x[2], reverse=True)

        return all_items
