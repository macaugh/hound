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
