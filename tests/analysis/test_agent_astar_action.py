# tests/analysis/test_agent_astar_action.py
import pytest
import json
from pathlib import Path
from analysis.agent_core import AutonomousAgent


def test_agent_astar_suggest_next_node(tmp_path):
    """Test agent can use A* to suggest next node to explore."""
    # Create mock graph and manifest
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()

    # Mock graph file with security-critical nodes
    graph_data = {
        'nodes': [
            {'id': 'func1', 'type': 'function', 'label': 'authenticate'},
            {'id': 'func2', 'type': 'function', 'label': 'helper'},
            {'id': 'func3', 'type': 'function', 'label': 'transfer'}
        ],
        'edges': [
            {'source_id': 'func1', 'target_id': 'func2', 'type': 'calls'},
            {'source_id': 'func2', 'target_id': 'func3', 'type': 'calls'}
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
        json.dump({'num_files': 3, 'repo_path': str(tmp_path)}, f)

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

    # Mock loaded graph (system graph already auto-loaded)
    # Just verify it's loaded
    assert agent.loaded_data['system_graph'] is not None

    # Get A* suggestion
    suggestion = agent._get_astar_suggestion()

    # Should suggest a high-priority node (authenticate or transfer)
    assert suggestion is not None
    assert 'node_id' in suggestion
    assert suggestion['node_id'] in ['func1', 'func3']  # High priority nodes
    assert suggestion['priority'] == 'high'
    assert 'reasoning' in suggestion
