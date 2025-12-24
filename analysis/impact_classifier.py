"""Impact classification for vulnerability hypotheses."""
from typing import Any


class ImpactClassifier:
    """Classify impact type: security vs compatibility vs quality."""

    # Pattern keywords for each category
    PATTERNS = {
        'security': [
            'theft', 'loss', 'unauthorized', 'bypass', 'manipulation',
            'drain', 'steal', 'lock', 'freeze', 'reentrancy',
            'overflow', 'underflow', 'access control'
        ],
        'compatibility': [
            'gas limit', 'does not work for', 'incompatible',
            'smart contract wallet', 'certain users cannot',
            'fails for', 'cannot be used by'
        ],
        'quality': [
            'dust', 'precision loss', 'rounding', 'wei-level',
            'unbounded array', 'pagination', 'view function',
            'gas optimization', 'code quality'
        ]
    }

    def classify(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """
        Classify hypothesis impact type.

        Args:
            hypothesis: Hypothesis dict with title, description, vulnerability_type

        Returns:
            {
                'category': 'security' | 'compatibility' | 'quality',
                'confidence': float,
                'reasoning': str,
                'disqualifying': bool  # True if not security
            }
        """
        # Combine text fields
        text = ' '.join([
            hypothesis.get('title', ''),
            hypothesis.get('description', ''),
            hypothesis.get('vulnerability_type', '')
        ]).lower()

        # Count matches per category and track matched keywords
        scores = {}
        matched_keywords = {}
        for category, keywords in self.PATTERNS.items():
            matches = [keyword for keyword in keywords if keyword in text]
            scores[category] = len(matches)
            matched_keywords[category] = matches

        # Determine category by highest score
        if scores['security'] > scores['compatibility'] and scores['security'] > scores['quality']:
            category = 'security'
            disqualifying = False
        elif scores['compatibility'] > scores['quality']:
            category = 'compatibility'
            disqualifying = True
        else:
            category = 'quality'
            disqualifying = True

        # Calculate confidence
        total_matches = sum(scores.values())
        confidence = scores[category] / max(total_matches, 1)

        # Build reasoning with matched keywords
        keywords_str = ', '.join(matched_keywords[category][:3]) if matched_keywords[category] else ''
        if category == 'security':
            reasoning = f"Security vulnerability (fund theft, unauthorized access, or manipulation)"
        elif category == 'compatibility':
            reasoning = f"Compatibility issue: {keywords_str}" if keywords_str else "Compatibility issue (feature doesn't work for some users)"
        else:
            reasoning = f"Quality issue: {keywords_str}" if keywords_str else "Quality issue (non-critical, no fund theft or loss)"

        return {
            'category': category,
            'confidence': confidence,
            'reasoning': reasoning,
            'disqualifying': disqualifying
        }
