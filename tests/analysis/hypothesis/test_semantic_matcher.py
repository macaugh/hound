import pytest
from analysis.hypothesis.semantic_matcher import SemanticMatcher


def test_semantic_matcher_initialization():
    """Test semantic matcher loads model successfully."""
    matcher = SemanticMatcher()

    assert matcher.model is not None
    assert matcher.threshold == 0.85  # Default threshold


def test_semantic_matcher_exact_duplicate():
    """Test matcher detects exact duplicates."""
    matcher = SemanticMatcher()

    hyp1 = "Missing authentication check in transfer function"
    hyp2 = "Missing authentication check in transfer function"

    similarity = matcher.compute_similarity(hyp1, hyp2)

    assert similarity > 0.95  # Should be nearly 1.0 for exact match


def test_semantic_matcher_semantic_duplicate():
    """Test matcher detects semantic duplicates with different wording."""
    matcher = SemanticMatcher()

    hyp1 = "Authentication bypass in withdraw function"
    hyp2 = "Missing auth check allows unauthorized withdrawals"

    similarity = matcher.compute_similarity(hyp1, hyp2)

    # Should be high similarity despite different wording
    assert similarity > 0.69


def test_semantic_matcher_different_vulnerabilities():
    """Test matcher distinguishes different vulnerability types."""
    matcher = SemanticMatcher()

    hyp1 = "Reentrancy vulnerability in transfer function"
    hyp2 = "Integer overflow in balance calculation"

    similarity = matcher.compute_similarity(hyp1, hyp2)

    # Should be low similarity - different issues
    assert similarity < 0.50
