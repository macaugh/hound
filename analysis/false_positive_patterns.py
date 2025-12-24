"""False positive pattern matching for vulnerability hypotheses."""
from typing import Any


class FalsePositivePatternMatcher:
    """Match against known false positive patterns."""

    PATTERNS = [
        {
            'name': 'admin_footgun',
            'indicators': ['onlyOwner', 'onlyAdmin', 'diamondCut', 'facet', 'upgrade'],
            'description': 'Only admin can trigger',
            'rule': 'If only admin can trigger, not a vulnerability',
            'severity': 'DISQUALIFY'
        },
        {
            'name': 'sybil_resistance',
            'indicators': [
                'one per address', 'multiple addresses', 'create new wallet',
                'ownership transfer', 'new owner', 'bypass one loan'
            ],
            'description': 'Users can create multiple addresses',
            'rule': 'Normal blockchain limitation, not vulnerability',
            'severity': 'DISQUALIFY'
        },
        {
            'name': 'mathematical_noise',
            'indicators': [
                'dust', 'wei', 'precision loss', 'division truncation',
                'rounding', '1-2 wei', 'integer division'
            ],
            'description': 'Wei-level precision loss',
            'rule': 'Economically irrelevant, all Solidity has this',
            'severity': 'DISQUALIFY'
        },
        {
            'name': 'design_limitation',
            'indicators': [
                'unbounded array', 'getAllLoans', 'view function',
                'too many users', 'gas limit', 'pagination'
            ],
            'description': 'Design choice with pagination available',
            'rule': 'Not exploitable if pagination exists',
            'severity': 'WARN'
        }
    ]

    def match(self, hypothesis: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Match hypothesis against false positive patterns.

        Args:
            hypothesis: Hypothesis dict with title, description
            analysis: Dict with permission_analysis, impact_classification, etc.

        Returns:
            {
                'matches': [pattern_name, ...],
                'confidence': float,
                'disqualifying': bool,
                'reasons': [str, ...]
            }
        """
        # Combine text fields
        text = ' '.join([
            hypothesis.get('title', ''),
            hypothesis.get('description', ''),
            hypothesis.get('vulnerability_type', '')
        ]).lower()

        matches = []
        reasons = []

        # Check each pattern
        for pattern in self.PATTERNS:
            # Check if any indicators present
            indicator_count = sum(1 for indicator in pattern['indicators'] if indicator.lower() in text)

            # Also check permission analysis
            if 'trigger_level' in analysis:
                if pattern['name'] == 'admin_footgun' and analysis['trigger_level'] == 'admin':
                    indicator_count += 2  # Boost admin footgun score

            # Match threshold
            if indicator_count > 0:
                matches.append(pattern['name'])
                reasons.append(f"{pattern['name']}: {pattern['rule']}")

        # Determine if disqualifying
        disqualifying = any(
            pattern['severity'] == 'DISQUALIFY'
            for pattern in self.PATTERNS
            if pattern['name'] in matches
        )

        # Calculate confidence
        confidence = min(1.0, len(matches) * 0.4)

        return {
            'matches': matches,
            'confidence': confidence,
            'disqualifying': disqualifying,
            'reasons': reasons
        }
