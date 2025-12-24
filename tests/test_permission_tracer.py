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

    def test_detect_require_owner(self):
        """Test detection of require(msg.sender == owner) pattern."""
        tracer = PermissionTracer()

        source_code = {
            'Contract.sol': '''
                function withdraw() public {
                    require(msg.sender == owner, "Not owner");
                    payable(owner).transfer(address(this).balance);
                }
            '''
        }

        hypothesis = {'title': 'Unauthorized withdrawal'}
        result = tracer.analyze_permissions(hypothesis, source_code)

        assert result['trigger_level'] == 'admin'
        assert result['disqualifying'] is True

    def test_detect_enforce_is_contract_owner(self):
        """Test detection of enforceIsContractOwner() pattern (Diamond)."""
        tracer = PermissionTracer()

        source_code = {
            'Facet.sol': '''
                function adminFunction() external {
                    DiamondStorageLib.enforceIsContractOwner();
                    // admin logic
                }
            '''
        }

        hypothesis = {'title': 'Admin bypass'}
        result = tracer.analyze_permissions(hypothesis, source_code)

        assert result['trigger_level'] == 'admin'
        assert 'enforceIsContractOwner' in result['modifiers']
        assert result['disqualifying'] is True

    def test_no_restrictions(self):
        """Test detection of no permission restrictions."""
        tracer = PermissionTracer()

        source_code = {
            'Contract.sol': '''
                function publicFunction() public {
                    // anyone can call
                }
            '''
        }

        hypothesis = {'title': 'Public vulnerability'}
        result = tracer.analyze_permissions(hypothesis, source_code)

        assert result['trigger_level'] == 'anyone'
        assert result['disqualifying'] is False
