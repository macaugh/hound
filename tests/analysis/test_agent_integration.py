# tests/analysis/test_agent_integration.py
import pytest
from pathlib import Path
from analysis.agent_core import AutonomousAgent


def test_agent_uses_astar_for_exploration(tmp_path):
    """Test that agent uses A* search when enabled."""
    # Create mock graph and manifest
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()

    # Mock graph file
    import json
    graph_data = {
        'nodes': [
            {'id': 'func1', 'type': 'function', 'label': 'authenticate'},
            {'id': 'func2', 'type': 'function', 'label': 'helper'}
        ],
        'edges': [
            {'source_id': 'func1', 'target_id': 'func2', 'type': 'calls'}
        ]
    }

    with open(graphs_dir / "knowledge_graphs.json", 'w') as f:
        json.dump({
            'graphs': {'TestGraph': str(graphs_dir / "graph_TestGraph.json")}
        }, f)

    with open(graphs_dir / "graph_TestGraph.json", 'w') as f:
        json.dump(graph_data, f)

    # Mock manifest
    with open(manifest_dir / "manifest.json", 'w') as f:
        json.dump({'num_files': 2, 'repo_path': str(tmp_path)}, f)

    # Create agent with A* enabled
    config = {
        'models': {
            'scout': {'model': 'gpt-4', 'provider': 'mock'},
            'strategist': {'model': 'gpt-4', 'provider': 'mock'}
        },
        'agent': {
            'use_astar_search': True
        }
    }

    agent = AutonomousAgent(
        graphs_metadata_path=graphs_dir / "knowledge_graphs.json",
        manifest_path=manifest_dir,
        agent_id="test_agent",
        config=config
    )

    # Verify A* search is enabled
    assert hasattr(agent, 'astar_search')
    assert agent.use_astar_search is True
