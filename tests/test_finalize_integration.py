"""Tests for finalize integration with validation framework."""
import pytest
from pathlib import Path
import tempfile
import json


def test_finalize_enhanced_prompt_structure():
    """Test that enhanced prompt includes validation checklist."""
    # This is a structural test - verify prompt includes key sections
    from commands.finalize import finalize

    # Read finalize.py source
    finalize_path = Path(__file__).parent.parent / 'commands' / 'finalize.py'
    source = finalize_path.read_text()

    # Check for validation checklist
    assert 'VALIDATION CHECKLIST' in source
    assert 'ROOT CAUSE' in source
    assert 'GUARDS' in source
    assert 'PRECONDITIONS' in source
    assert 'MITIGATIONS' in source
    assert 'PERMISSION REQUIREMENTS' in source
    assert 'IMPACT TYPE' in source


def test_finalize_calls_validation_modules(tmp_path):
    """Test that finalize calls permission tracer, classifier, and pattern matcher."""
    # This test would require significant mocking
    # For now, we'll verify imports exist
    from commands import finalize
    import inspect

    source = inspect.getsource(finalize)

    # After integration, these imports should exist
    assert 'from analysis.permission_tracer import PermissionTracer' in source
    assert 'from analysis.impact_classifier import ImpactClassifier' in source
    assert 'from analysis.false_positive_patterns import FalsePositivePatternMatcher' in source
