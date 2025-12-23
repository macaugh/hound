"""Attention-based context selection for agent.

Implements transformer-style attention mechanism using sentence embeddings.
Computes attention scores between investigation goal and context parts.

NOTE: This module is being developed using Test-Driven Development (TDD).
All implementations are driven by tests in tests/test_attention_*.py

DESIGN DECISION: Uses raw cosine similarity (not softmax) for attention scores.
Rationale: Absolute relevance is more useful than relative scores for threshold filtering.
Scores represent semantic similarity in [0, 1] range, interpretable across different contexts.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AttentionScorer:
    """Compute attention scores for context selection using semantic similarity."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize scorer with sentence transformer model.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, any] = {}

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
        # Encode query and contexts
        query_emb = self.model.encode(query, convert_to_numpy=True)
        context_embs = np.array([
            self.model.encode(ctx, convert_to_numpy=True)
            for ctx in contexts
        ])

        # Compute cosine similarities
        scores = cosine_similarity(
            query_emb.reshape(1, -1),
            context_embs
        )[0]

        # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
        normalized_scores = (scores + 1.0) / 2.0

        return normalized_scores
