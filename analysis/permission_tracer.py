"""Permission analysis for vulnerability hypotheses."""
from typing import Any


class PermissionTracer:
    """Trace access control to determine who can trigger vulnerability."""

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
        reasoning = 'No permission restrictions detected'

        # Detect common Solidity modifiers
        if 'onlyOwner' in all_code:
            trigger_level = 'admin'
            modifiers.append('onlyOwner')
            disqualifying = True
            reasoning = 'Function restricted to only owner via onlyOwner modifier'

        return {
            'trigger_level': trigger_level,
            'modifiers': modifiers,
            'reasoning': reasoning,
            'disqualifying': disqualifying
        }
