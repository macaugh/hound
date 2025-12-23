"""Tests for AttentionFilter class.

Following TDD: Tests written BEFORE implementation.
Tests specify correct behavior, especially fixing bugs from code review:
- Critical sections should use ACTUAL scores (not hardcoded 1.0)
- top_k should NOT count critical sections toward limit
- Threshold filtering should work reliably
"""

import pytest
import numpy as np
from unittest.mock import Mock


def test_filter_includes_items_above_threshold():
    """Filter should include only items with score >= threshold."""
    from analysis.context.attention import AttentionScorer, AttentionFilter

    # Create scorer and filter
    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.7)

    query = "Find authentication vulnerabilities"
    contexts = [
        ("HIGH_RELEVANCE", "Authentication bypass allows unauthorized access"),
        ("MEDIUM_RELEVANCE", "Helper function validates user permissions"),
        ("LOW_RELEVANCE", "Logging utility formats timestamps")
    ]

    filtered = filter.filter_contexts(query, contexts, critical_labels=set())

    # Should only include high-relevance item (score > 0.7)
    labels = [f[0] for f in filtered]
    assert "HIGH_RELEVANCE" in labels
    assert "LOW_RELEVANCE" not in labels


def test_filter_excludes_items_below_threshold():
    """Filter should exclude items with score < threshold."""
    from analysis.context.attention import AttentionScorer, AttentionFilter

    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.8)

    query = "Security vulnerability in payment processing"
    contexts = [
        ("RELATED", "Payment processing has SQL injection vulnerability"),
        ("UNRELATED", "The weather is sunny today")
    ]

    filtered = filter.filter_contexts(query, contexts, critical_labels=set())

    # Unrelated should be filtered out
    labels = [f[0] for f in filtered]
    assert "UNRELATED" not in labels


def test_critical_sections_use_actual_scores():
    """CRITICAL: Critical sections should use actual attention scores, not hardcoded 1.0.

    This fixes bug from code review where critical sections were given hardcoded 1.0 score,
    misrepresenting their actual relevance.
    """
    from analysis.context.attention import AttentionScorer, AttentionFilter

    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.5)

    query = "Find SQL injection vulnerabilities"
    contexts = [
        ("CRITICAL_BUT_UNRELATED", "The weather is sunny today"),  # Critical but low relevance
        ("NON_CRITICAL_RELATED", "SQL injection in database query")  # Not critical but high relevance
    ]

    filtered = filter.filter_contexts(query, contexts, critical_labels={"CRITICAL_BUT_UNRELATED"})

    # Find the critical section
    critical_item = [f for f in filtered if f[0] == "CRITICAL_BUT_UNRELATED"][0]
    _, _, score = critical_item

    # Score should be actual relevance (~0.5 for unrelated), NOT 1.0
    assert 0.4 < score < 0.6, f"Critical section should have actual score ~0.5, got {score}"
    assert score != 1.0, "Critical section should NOT have hardcoded 1.0 score"


def test_critical_sections_always_included():
    """Critical sections must be included even if below threshold."""
    from analysis.context.attention import AttentionScorer, AttentionFilter

    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.9)  # Very high threshold

    query = "Find reentrancy vulnerabilities"
    contexts = [
        ("INVESTIGATION_GOAL", "Project context and background info"),  # Critical but not highly relevant
        ("HIGH_RELEVANCE", "Reentrancy attack in withdraw function")
    ]

    filtered = filter.filter_contexts(query, contexts, critical_labels={"INVESTIGATION_GOAL"})

    # Critical section must be included even with low score
    labels = [f[0] for f in filtered]
    assert "INVESTIGATION_GOAL" in labels


def test_top_k_returns_correct_count_excluding_critical():
    """CRITICAL: top_k should return top K items PLUS all critical sections.

    This fixes bug from code review where critical sections counted toward top_k limit.
    If top_k=3 and 2 items are critical, should return 3 regular + 2 critical = 5 total.
    """
    from analysis.context.attention import AttentionScorer, AttentionFilter

    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.0, top_k=2)

    query = "Security analysis"
    contexts = [
        ("CRITICAL_1", "Critical context one"),
        ("CRITICAL_2", "Critical context two"),
        ("REGULAR_A", "Security vulnerability in authentication"),
        ("REGULAR_B", "Security issue in authorization"),
        ("REGULAR_C", "Security flaw in validation"),
        ("REGULAR_D", "Security problem in logging")
    ]

    filtered = filter.filter_contexts(
        query, contexts,
        critical_labels={"CRITICAL_1", "CRITICAL_2"}
    )

    # Should return: 2 critical + top 2 regular = 4 total
    assert len(filtered) == 4, f"Expected 4 items (2 critical + 2 regular), got {len(filtered)}"

    # Critical sections must be included
    labels = [f[0] for f in filtered]
    assert "CRITICAL_1" in labels
    assert "CRITICAL_2" in labels

    # Should have 2 regular items (highest scoring)
    regular_items = [f for f in filtered if f[0] not in {"CRITICAL_1", "CRITICAL_2"}]
    assert len(regular_items) == 2


def test_results_sorted_by_score_descending():
    """Results should be sorted with highest scores first."""
    from analysis.context.attention import AttentionScorer, AttentionFilter

    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.0)

    query = "Authentication security"
    contexts = [
        ("A", "Authentication bypass vulnerability"),  # High score
        ("B", "Random unrelated text"),  # Low score
        ("C", "Authorization security check")  # Medium-high score
    ]

    filtered = filter.filter_contexts(query, contexts, critical_labels=set())

    # Scores should be in descending order
    scores = [f[2] for f in filtered]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"


def test_empty_contexts_returns_empty():
    """Empty input should return empty output."""
    from analysis.context.attention import AttentionScorer, AttentionFilter

    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.5)

    filtered = filter.filter_contexts("query", [], critical_labels=set())

    assert filtered == []
