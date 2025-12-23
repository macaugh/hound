"""Semantic similarity matching for hypothesis deduplication."""
from __future__ import annotations

import threading
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityMatcher(Protocol):
    """Protocol for semantic similarity matchers."""
    threshold: float

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        ...


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
        self._cache_lock = threading.Lock()  # Thread-safe cache access

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
        # Check cache with lock
        with self._cache_lock:
            if text in self._cache:
                return self._cache[text]

        # Compute embedding outside lock (expensive operation)
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Update cache with lock
        with self._cache_lock:
            self._cache[text] = embedding

            # Limit cache size - evict to 80% capacity when exceeding limit
            if len(self._cache) > 1000:
                target_size = 800
                to_remove = len(self._cache) - target_size
                for _ in range(to_remove):
                    # Remove oldest entry (FIFO - dicts maintain insertion order)
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


def compute_node_overlap(nodes1: list[str], nodes2: list[str]) -> float:
    """Compute Jaccard similarity between two node lists.

    Args:
        nodes1: First node list
        nodes2: Second node list

    Returns:
        Overlap ratio (0-1)
    """
    if not nodes1 or not nodes2:
        return 0.0

    set1 = set(nodes1)
    set2 = set(nodes2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def is_duplicate_hypothesis(new_hyp: dict,
                            existing_hyps: list[dict],
                            matcher: SemanticMatcher,
                            node_overlap_threshold: float = 0.3) -> tuple[bool, dict | None]:
    """Check if hypothesis is duplicate considering both text and nodes.

    Args:
        new_hyp: New hypothesis dict with 'title' and 'node_refs'
        existing_hyps: List of existing hypothesis dicts
        matcher: SemanticMatcher instance
        node_overlap_threshold: Minimum node overlap to consider (0-1)

    Returns:
        Tuple of (is_duplicate, matched_hypothesis)
    """
    new_title = new_hyp.get('title', '')
    new_nodes = new_hyp.get('node_refs', [])

    for existing in existing_hyps:
        existing_title = existing.get('title', '')
        existing_nodes = existing.get('node_refs', [])

        # Compute text similarity
        text_sim = matcher.compute_similarity(new_title, existing_title)

        # Compute node overlap
        node_overlap = compute_node_overlap(new_nodes, existing_nodes)

        # Duplicate if BOTH conditions met:
        # 1. High text similarity (above threshold)
        # 2. Significant node overlap (above threshold)
        if text_sim >= matcher.threshold and node_overlap >= node_overlap_threshold:
            return True, existing

    return False, None


class DummyMatcher:
    """Fallback matcher that never matches (for graceful degradation)."""
    threshold: float = 1.0  # Set impossibly high threshold

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Always return 0.0 similarity."""
        return 0.0
