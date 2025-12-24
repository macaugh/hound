"""Permission analysis for vulnerability hypotheses."""
import re
from typing import Any


class PermissionTracer:
    """Trace access control to determine who can trigger vulnerability."""

    # Solidity permission patterns
    ADMIN_MODIFIERS = [
        'onlyOwner',
        'onlyAdmin',
        'onlyGovernance',
        'onlyController',
        'onlyRole'
    ]

    ADMIN_PATTERNS = [
        r'require\s*\(\s*msg\.sender\s*==\s*owner',
        r'require\s*\(\s*msg\.sender\s*==\s*admin',
        r'enforceIsContractOwner\s*\(',
        r'diamondCut\s*\(',
        r'onlyOwner\s+modifier',
    ]

    def analyze_permissions(self, hypothesis: dict[str, Any], source_files: dict[str, str]) -> dict[str, Any]:
        """
        Analyze source code to determine permission requirements.

        Args:
            hypothesis: Hypothesis dict with title, description
            source_files: Dict of filename -> source code

        Returns:
            {
                'trigger_level': 'anyone' | 'user' | 'admin' | 'owner',
                'modifiers': [list of detected modifiers],
                'reasoning': 'explanation',
                'disqualifying': bool  # True if admin-only
            }
        """
        # Combine all source code
        all_code = '\n'.join(source_files.values())

        # Default values
        trigger_level = 'anyone'
        modifiers = []
        disqualifying = False
        reasoning_parts = []

        # Check for modifier keywords
        for modifier in self.ADMIN_MODIFIERS:
            if modifier in all_code:
                trigger_level = 'admin'
                modifiers.append(modifier)
                disqualifying = True
                # Add space for readability (e.g., "only Owner" instead of "onlyOwner")
                readable_modifier = re.sub(r'([a-z])([A-Z])', r'\1 \2', modifier).lower()
                reasoning_parts.append(f"Uses {readable_modifier} modifier")

        # Check for require/enforce patterns
        for pattern in self.ADMIN_PATTERNS:
            matches = re.findall(pattern, all_code, re.IGNORECASE)
            if matches:
                trigger_level = 'admin'
                # Extract pattern name
                pattern_name = pattern.split('\\')[0].replace('r\'', '')
                if pattern_name not in modifiers:
                    modifiers.append(pattern_name)
                disqualifying = True
                reasoning_parts.append(f"Contains admin check: {matches[0][:50]}")

        # Build final reasoning
        if reasoning_parts:
            reasoning = '; '.join(reasoning_parts[:3])  # Limit to 3 reasons
        else:
            reasoning = 'No permission restrictions detected in source code'

        return {
            'trigger_level': trigger_level,
            'modifiers': modifiers,
            'reasoning': reasoning,
            'disqualifying': disqualifying
        }
