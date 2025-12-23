"""Hypothesis management and deduplication."""
from .semantic_matcher import DummyMatcher, SemanticMatcher, SimilarityMatcher

__all__ = ["SemanticMatcher", "SimilarityMatcher", "DummyMatcher"]
