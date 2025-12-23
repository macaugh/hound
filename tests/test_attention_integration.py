"""Integration tests for attention filtering in agent context.

Following TDD: Write integration test BEFORE implementing agent changes.
Tests prove attention filtering works end-to-end in agent.
"""

import pytest
from pathlib import Path
import tempfile
import json


def test_attention_filtering_reduces_context_size():
    """Integration test: Attention filtering reduces agent context without losing critical info.

    This is the KEY test proving the feature solves the timeout problem.
    """
    # Skip if dependencies not available (integration test)
    pytest.importorskip("sentence_transformers")

    from analysis.agent_core import AutonomousAgent

    # Create temporary project structure
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "test_project"
        project_dir.mkdir()

        graphs_dir = project_dir / "graphs"
        graphs_dir.mkdir()

        # Create minimal graphs metadata
        graphs_metadata = graphs_dir / "graphs_metadata.json"
        graphs_metadata.write_text(json.dumps({
            "SystemArchitecture": {
                "description": "System architecture graph",
                "nodes": [],
                "edges": []
            }
        }))

        # Create minimal manifest
        manifest_path = project_dir / "source" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "cards": []
        }))

        # Test WITH attention filtering
        config_with_attention = {
            'models': {
                'scout': {'model': 'mock-model', 'provider': 'mock'}
            },
            'agent': {
                'use_attention_context': True,
                'attention_threshold': 0.6  # Moderate threshold
            }
        }

        agent_with = AutonomousAgent(
            graphs_metadata_path=graphs_metadata,
            manifest_path=manifest_path,
            agent_id="test_agent_with",
            config=config_with_attention
        )

        # Set investigation goal
        agent_with.investigation_goal = "Find reentrancy vulnerabilities in transfer functions"

        # Add lots of context (some relevant, some not)
        agent_with.memory_notes = [
            "Found potential reentrancy in withdraw",
            "Transfer function modifies state",
            "Weather is sunny today",  # Irrelevant
            "Recipe for cookies",  # Irrelevant
            "Reentrancy guard missing",
            "Color preferences saved",  # Irrelevant
            "State changes before external call",
            "User preferences updated",  # Irrelevant
        ]

        # Build context with attention
        context_with_attention = agent_with._build_context()
        tokens_with = agent_with._count_tokens(context_with_attention)

        # Test WITHOUT attention filtering
        config_without_attention = {
            'models': {
                'scout': {'model': 'mock-model', 'provider': 'mock'}
            },
            'agent': {
                'use_attention_context': False
            }
        }

        agent_without = AutonomousAgent(
            graphs_metadata_path=graphs_metadata,
            manifest_path=manifest_path,
            agent_id="test_agent_without",
            config=config_without_attention
        )

        agent_without.investigation_goal = "Find reentrancy vulnerabilities in transfer functions"
        agent_without.memory_notes = agent_with.memory_notes  # Same notes

        # Build context without attention
        context_without_attention = agent_without._build_context()
        tokens_without = agent_without._count_tokens(context_without_attention)

        # Attention filtering should reduce token count
        reduction = (tokens_without - tokens_with) / tokens_without if tokens_without > 0 else 0

        print(f"\nContext size comparison:")
        print(f"  Without attention: {tokens_without} tokens")
        print(f"  With attention: {tokens_with} tokens")
        print(f"  Reduction: {reduction*100:.1f}%")

        # Should reduce tokens (goal: 20-40% reduction with moderate threshold)
        assert tokens_with < tokens_without, "Attention filtering should reduce context size"

        # Critical sections must still be present (labels use underscores)
        assert "INVESTIGATION_GOAL" in context_with_attention
        assert "reentrancy vulnerabilities" in context_with_attention

        # High-relevance notes should be present
        assert "reentrancy" in context_with_attention.lower()

        # Low-relevance notes likely filtered (but not guaranteed with this threshold)
        # Just verify the filtering mechanism worked
        assert reduction > 0.05, f"Should see at least 5% reduction, got {reduction*100:.1f}%"


def test_critical_sections_always_preserved():
    """Integration test: Critical sections preserved regardless of relevance."""
    pytest.importorskip("sentence_transformers")

    from analysis.agent_core import AutonomousAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "test_project"
        project_dir.mkdir()

        graphs_dir = project_dir / "graphs"
        graphs_dir.mkdir()

        graphs_metadata = graphs_dir / "graphs_metadata.json"
        graphs_metadata.write_text(json.dumps({
            "SystemArchitecture": {"description": "System", "nodes": [], "edges": []}
        }))

        manifest_path = project_dir / "source" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"cards": []}))

        config = {
            'models': {'scout': {'model': 'mock-model', 'provider': 'mock'}},
            'agent': {
                'use_attention_context': True,
                'attention_threshold': 0.9  # Very high - filters almost everything
            }
        }

        agent = AutonomousAgent(
            graphs_metadata_path=graphs_metadata,
            manifest_path=manifest_path,
            agent_id="test_critical",
            config=config
        )

        # Investigation goal completely unrelated to memory
        agent.investigation_goal = "Find SQL injection vulnerabilities"
        agent.memory_notes = [
            "Weather forecast says rain",
            "Recipe needs three eggs",
            "Movie starts at 8pm"
        ]

        context = agent._build_context()

        # Critical sections MUST be present even though unrelated (labels use underscores)
        assert "INVESTIGATION_GOAL" in context
        assert "SQL injection" in context

        # Even with high threshold, agent should function (has critical sections)
        assert len(context) > 50, "Context should contain critical sections"


def test_attention_initialization_defaults():
    """Test that attention filtering defaults to disabled."""
    from analysis.agent_core import AutonomousAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "test_project"
        project_dir.mkdir()

        graphs_dir = project_dir / "graphs"
        graphs_dir.mkdir()

        graphs_metadata = graphs_dir / "graphs_metadata.json"
        graphs_metadata.write_text(json.dumps({
            "SystemArchitecture": {"description": "System", "nodes": [], "edges": []}
        }))

        manifest_path = project_dir / "source" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"cards": []}))

        # No attention config
        config = {
            'models': {'scout': {'model': 'mock-model', 'provider': 'mock'}}
        }

        agent = AutonomousAgent(
            graphs_metadata_path=graphs_metadata,
            manifest_path=manifest_path,
            agent_id="test_default",
            config=config
        )

        # Should default to disabled
        assert hasattr(agent, 'use_attention')
        assert agent.use_attention == False, "Attention filtering should default to disabled"
