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


def test_semantic_matcher_with_node_overlap():
    """Test that node overlap is considered in duplicate detection."""
    from analysis.hypothesis.semantic_matcher import is_duplicate_hypothesis

    # New hypothesis
    new_hyp = {
        'title': 'Missing access control in transfer',
        'node_refs': ['func_transfer', 'func_validate']
    }

    # Existing hypothesis - similar text, overlapping nodes
    existing1 = {
        'title': 'Authorization bypass in transfer function',
        'node_refs': ['func_transfer', 'func_authorize']  # 50% overlap
    }

    # Existing hypothesis - similar text, NO overlapping nodes
    existing2 = {
        'title': 'Missing authorization in withdrawal',
        'node_refs': ['func_withdraw', 'func_check']  # 0% overlap
    }

    matcher = SemanticMatcher(threshold=0.67)  # Lower threshold for testing

    # Should be duplicate due to high similarity AND node overlap
    is_dup1, _ = is_duplicate_hypothesis(new_hyp, [existing1], matcher)
    assert is_dup1 is True

    # Should NOT be duplicate - different nodes despite similar text
    is_dup2, _ = is_duplicate_hypothesis(new_hyp, [existing2], matcher)
    assert is_dup2 is False


def test_semantic_matcher_node_overlap_calculation():
    """Test node overlap calculation."""
    from analysis.hypothesis.semantic_matcher import compute_node_overlap

    nodes1 = ['a', 'b', 'c']
    nodes2 = ['b', 'c', 'd']

    overlap = compute_node_overlap(nodes1, nodes2)

    # Overlap = intersection / union = 2 / 4 = 0.5
    assert overlap == pytest.approx(0.5, abs=0.01)
