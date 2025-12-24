"""Tests for permission tracer module."""
import pytest
from analysis.permission_tracer import PermissionTracer


class TestPermissionTracer:
    """Test permission analysis functionality."""

    def test_detect_onlyowner_modifier(self):
        """Test detection of onlyOwner modifier in Solidity."""
        tracer = PermissionTracer()

        source_code = {
            'Contract.sol': '''
                function withdraw() public onlyOwner {
                    payable(owner).transfer(address(this).balance);
                }
            '''
        }

        hypothesis = {
            'title': 'Unauthorized withdrawal',
            'description': 'Anyone can withdraw funds'
        }

        result = tracer.analyze_permissions(hypothesis, source_code)

        assert result['trigger_level'] == 'admin'
        assert 'onlyOwner' in result['modifiers']
        assert result['disqualifying'] is True
        assert 'only owner' in result['reasoning'].lower()
