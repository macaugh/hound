"""Tests for AttentionScorer class.

Following TDD: Tests written BEFORE implementation.
Each test specifies behavior, then we implement to pass.
"""

import pytest
import numpy as np


def test_scorer_initializes_with_default_model():
    """AttentionScorer can be initialized with default model."""
    from analysis.context.attention import AttentionScorer

    scorer = AttentionScorer()

    # Should have a model loaded
    assert scorer.model is not None
    # Should have empty cache initially
    assert len(scorer._cache) == 0
    # Should store model name for verification
    assert scorer.model_name == 'all-MiniLM-L6-v2'


def test_compute_attention_scores_returns_correct_array_length():
    """compute_attention_scores returns array matching context count."""
    from analysis.context.attention import AttentionScorer

    scorer = AttentionScorer()
    query = "Find authentication vulnerabilities"
    contexts = [
        "Authentication bypass in login",
        "Helper function for formatting",
        "Authorization check missing"
    ]

    scores = scorer.compute_attention_scores(query, contexts)

    # Should return array with one score per context
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(contexts)
    assert scores.shape == (3,)


def test_identical_query_and_context_scores_high():
    """Identical text should get high similarity score."""
    from analysis.context.attention import AttentionScorer

    scorer = AttentionScorer()
    text = "Check for reentrancy vulnerabilities in transfer function"

    scores = scorer.compute_attention_scores(text, [text])

    # Identical text should score very high (close to 1.0 for cosine similarity)
    assert scores[0] > 0.9


def test_unrelated_query_and_context_scores_low():
    """Completely unrelated text should get baseline/neutral similarity score."""
    from analysis.context.attention import AttentionScorer

    scorer = AttentionScorer()
    query = "Find SQL injection vulnerabilities in database queries"
    contexts = [
        "The weather today is sunny with clear skies",
        "Recipe for chocolate chip cookies with butter"
    ]

    scores = scorer.compute_attention_scores(query, contexts)

    # Unrelated text should cluster around 0.5 (neutral/baseline)
    # Should be significantly lower than high-similarity scores (>0.9)
    assert all(0.4 < score < 0.6 for score in scores)  # Baseline range
    assert all(score < 0.7 for score in scores)  # Clearly not similar


def test_encoding_cache_reduces_computation():
    """Cache should store encodings and reuse them for repeated text."""
    from analysis.context.attention import AttentionScorer
    from unittest.mock import patch

    scorer = AttentionScorer()
    query = "Check for reentrancy vulnerabilities"
    context1 = "Transfer function allows reentrancy"
    context2 = "Helper function validates input"

    # First call should encode all unique texts
    with patch.object(scorer.model, 'encode', wraps=scorer.model.encode) as mock_encode:
        scorer.compute_attention_scores(query, [context1, context2])

        # Should encode query once + 2 unique contexts
        assert mock_encode.call_count == 3

    # Second call with same texts should use cache
    with patch.object(scorer.model, 'encode', wraps=scorer.model.encode) as mock_encode:
        scorer.compute_attention_scores(query, [context1, context2])

        # Should not encode anything (all cached)
        assert mock_encode.call_count == 0


def test_cache_eviction_at_limit():
    """Cache should evict old items when exceeding size limit."""
    from analysis.context.attention import AttentionScorer

    scorer = AttentionScorer()

    # Fill cache to limit (500 items)
    for i in range(500):
        text = f"Test text number {i}"
        scorer.compute_attention_scores(text, [text])

    # Cache should be at limit
    assert len(scorer._cache) == 500

    # Add one more unique item
    scorer.compute_attention_scores("New unique text", ["Another unique text"])

    # Cache should still be at limit (evicted oldest)
    assert len(scorer._cache) <= 500
