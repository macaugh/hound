"""Tests for impact classifier module."""
import pytest
from analysis.impact_classifier import ImpactClassifier


class TestImpactClassifier:
    """Test impact classification functionality."""

    def test_classify_security_vulnerability(self):
        """Test classification of security vulnerability."""
        classifier = ImpactClassifier()

        hypothesis = {
            'title': 'Unauthorized fund theft',
            'description': 'Attacker can steal all funds via reentrancy',
            'vulnerability_type': 'reentrancy'
        }

        result = classifier.classify(hypothesis)

        assert result['category'] == 'security'
        assert result['disqualifying'] is False
        assert result['confidence'] > 0.7

    def test_classify_compatibility_issue(self):
        """Test classification of compatibility issue."""
        classifier = ImpactClassifier()

        hypothesis = {
            'title': 'Smart contract wallets cannot claim',
            'description': 'Gas limit of 5000 is insufficient for smart contract wallets',
            'vulnerability_type': 'gas_limit'
        }

        result = classifier.classify(hypothesis)

        assert result['category'] == 'compatibility'
        assert result['disqualifying'] is True
        assert 'gas limit' in result['reasoning'].lower()

    def test_classify_quality_issue(self):
        """Test classification of quality/dust issue."""
        classifier = ImpactClassifier()

        hypothesis = {
            'title': 'Wei-level precision loss',
            'description': 'Integer division causes dust loss of 1-2 wei per transaction',
            'vulnerability_type': 'precision_loss'
        }

        result = classifier.classify(hypothesis)

        assert result['category'] == 'quality'
        assert result['disqualifying'] is True
