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
