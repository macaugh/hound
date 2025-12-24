"""End-to-end tests for validation framework using DeltaPrime examples."""
import pytest
from analysis.permission_tracer import PermissionTracer
from analysis.impact_classifier import ImpactClassifier
from analysis.false_positive_patterns import FalsePositivePatternMatcher


class TestValidationE2E:
    """End-to-end validation tests with real-world examples."""

    def test_deltaprime_admin_footgun(self):
        """Test DeltaPrime Report 2: Liquidator DoS (admin foot-gun)."""
        hypothesis = {
            'title': 'Liquidator Check External Call DoS',
            'description': 'If liquidator facet is removed via diamondCut, liquidations fail',
            'vulnerability_type': 'dos'
        }

        source_code = {
            'DiamondCutFacet.sol': '''
                function diamondCut(FacetCut[] calldata _diamondCut, address _init, bytes calldata _calldata)
                    external virtual override paused {
                    require(address(this) == 0x968f..., "This can be called only on the DiamondBeacon contract");
                    DiamondStorageLib.enforceIsContractOwner();
                    DiamondStorageLib.diamondCut(_diamondCut, _init, _calldata);
                }
            '''
        }

        # Permission analysis
        tracer = PermissionTracer()
        perm_result = tracer.analyze_permissions(hypothesis, source_code)
        assert perm_result['trigger_level'] == 'admin'
        assert perm_result['disqualifying'] is True

        # Pattern matching
        matcher = FalsePositivePatternMatcher()
        pattern_result = matcher.match(hypothesis, perm_result)
        assert 'admin_footgun' in pattern_result['matches']
        assert pattern_result['disqualifying'] is True

    def test_deltaprime_sybil_resistance(self):
        """Test DeltaPrime Report 8: hasNoLoan bypass (Sybil resistance)."""
        hypothesis = {
            'title': 'hasNoLoan Bypass via Ownership Transfer',
            'description': 'User can create multiple loans by transferring ownership to new addresses',
            'vulnerability_type': 'access_control'
        }

        source_code = {
            'SmartLoansFactory.sol': '''
                function changeOwnership(address _newOwner) public {
                    address loan = msg.sender;
                    address oldOwner = loansToOwners[loan];
                    require(oldOwner != address(0), "Only a SmartLoan can change its owner");
                    require(!_hasLoan(_newOwner), "New owner already has a loan");
                    ownersToLoans[oldOwner] = address(0);  // Correct: old owner now has no loan
                    ownersToLoans[_newOwner] = loan;
                    loansToOwners[loan] = _newOwner;
                }
            '''
        }

        # Permission analysis
        tracer = PermissionTracer()
        perm_result = tracer.analyze_permissions(hypothesis, source_code)
        assert perm_result['trigger_level'] == 'anyone'  # No special permissions
        assert perm_result['disqualifying'] is False

        # Pattern matching
        matcher = FalsePositivePatternMatcher()
        pattern_result = matcher.match(hypothesis, perm_result)
        assert 'sybil_resistance' in pattern_result['matches']
        assert pattern_result['disqualifying'] is True

    def test_deltaprime_mathematical_noise(self):
        """Test DeltaPrime Report 5: Integer division dust loss."""
        hypothesis = {
            'title': 'Integer Division Reward Dust Loss',
            'description': 'Division truncation in rewardRate calculation loses 1-2 wei per duration',
            'vulnerability_type': 'precision_loss'
        }

        source_code = {
            'DepositRewarderNative.sol': '''
                function notifyRewardAmount() external payable onlyOwner updateReward(address(0)) {
                    if (block.timestamp >= finishAt) {
                        rewardRate = msg.value / duration;  // Integer division
                    } else {
                        uint256 remainingRewards = (finishAt - block.timestamp) * rewardRate;
                        rewardRate = (msg.value + remainingRewards) / duration;
                    }
                }
            '''
        }

        # Impact classification
        classifier = ImpactClassifier()
        impact_result = classifier.classify(hypothesis)
        assert impact_result['category'] == 'quality'
        assert impact_result['disqualifying'] is True

        # Pattern matching
        matcher = FalsePositivePatternMatcher()
        pattern_result = matcher.match(hypothesis, {})
        assert 'mathematical_noise' in pattern_result['matches']
        assert pattern_result['disqualifying'] is True

    def test_deltaprime_real_vulnerability(self):
        """Test that real vulnerabilities are NOT rejected."""
        hypothesis = {
            'title': 'Reentrancy in withdraw function',
            'description': 'Attacker can drain all funds via reentrancy attack',
            'vulnerability_type': 'reentrancy'
        }

        source_code = {
            'Vault.sol': '''
                function withdraw() public {
                    uint256 amount = balances[msg.sender];
                    (bool success, ) = msg.sender.call{value: amount}("");
                    require(success);
                    balances[msg.sender] = 0;  // State update after external call!
                }
            '''
        }

        # Permission analysis
        tracer = PermissionTracer()
        perm_result = tracer.analyze_permissions(hypothesis, source_code)
        assert perm_result['trigger_level'] == 'anyone'
        assert perm_result['disqualifying'] is False

        # Impact classification
        classifier = ImpactClassifier()
        impact_result = classifier.classify(hypothesis)
        assert impact_result['category'] == 'security'
        assert impact_result['disqualifying'] is False

        # Pattern matching
        matcher = FalsePositivePatternMatcher()
        pattern_result = matcher.match(hypothesis, perm_result)
        assert pattern_result['disqualifying'] is False
