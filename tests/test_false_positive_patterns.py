"""Tests for false positive pattern matcher."""
import pytest
from analysis.false_positive_patterns import FalsePositivePatternMatcher


class TestFalsePositivePatternMatcher:
    """Test false positive pattern matching."""

    def test_match_admin_footgun(self):
        """Test detection of admin foot-gun pattern."""
        matcher = FalsePositivePatternMatcher()

        hypothesis = {
            'title': 'Liquidator DoS via facet removal',
            'description': 'If liquidator facet is removed, liquidations fail'
        }

        permission_analysis = {
            'trigger_level': 'admin',
            'modifiers': ['onlyOwner', 'diamondCut']
        }

        result = matcher.match(hypothesis, permission_analysis)

        assert 'admin_footgun' in result['matches']
        assert result['disqualifying'] is True

    def test_match_sybil_resistance(self):
        """Test detection of Sybil resistance pattern."""
        matcher = FalsePositivePatternMatcher()

        hypothesis = {
            'title': 'hasNoLoan bypass via ownership transfer',
            'description': 'User can create multiple loans by transferring ownership to new addresses'
        }

        permission_analysis = {
            'trigger_level': 'anyone'
        }

        result = matcher.match(hypothesis, permission_analysis)

        assert 'sybil_resistance' in result['matches']
        assert result['disqualifying'] is True

    def test_match_mathematical_noise(self):
        """Test detection of mathematical noise pattern."""
        matcher = FalsePositivePatternMatcher()

        hypothesis = {
            'title': 'Integer division dust loss',
            'description': 'Division truncation causes 1-2 wei loss per transaction'
        }

        permission_analysis = {
            'trigger_level': 'anyone'
        }

        result = matcher.match(hypothesis, permission_analysis)

        assert 'mathematical_noise' in result['matches']
        assert result['disqualifying'] is True

    def test_no_match(self):
        """Test no false positive pattern matched."""
        matcher = FalsePositivePatternMatcher()

        hypothesis = {
            'title': 'Reentrancy in withdraw function',
            'description': 'Attacker can drain funds via reentrancy'
        }

        permission_analysis = {
            'trigger_level': 'anyone'
        }

        result = matcher.match(hypothesis, permission_analysis)

        assert result['matches'] == []
        assert result['disqualifying'] is False
