# tests/analysis/search/test_astar.py
import pytest
from analysis.search.astar import SearchNode, SearchState, SecurityHeuristic, AStarSearch


def test_search_node_creation():
    """Test creating a search node with basic attributes."""
    node = SearchNode(
        node_id="func_transfer",
        graph_name="SystemArchitecture",
        g_score=0.0,
        h_score=0.5,
        parent=None
    )

    assert node.node_id == "func_transfer"
    assert node.graph_name == "SystemArchitecture"
    assert node.g_score == 0.0
    assert node.h_score == 0.5
    assert node.f_score == 0.5  # f = g + h
    assert node.parent is None


def test_search_node_comparison():
    """Test that nodes are compared by f_score for priority queue."""
    node1 = SearchNode("node1", "graph", g_score=1.0, h_score=2.0, parent=None)
    node2 = SearchNode("node2", "graph", g_score=0.5, h_score=2.0, parent=None)

    # node2 should have lower f_score (2.5 vs 3.0)
    assert node2 < node1


def test_search_state_initialization():
    """Test search state tracks explored nodes and frontier."""
    state = SearchState()

    assert len(state.explored) == 0
    assert len(state.frontier) == 0
    assert len(state.node_scores) == 0


def test_security_heuristic_high_risk_patterns():
    """Test heuristic identifies high-risk security patterns."""
    heuristic = SecurityHeuristic()

    # Mock node with high-risk patterns
    node_data = {
        'id': 'func_authenticate',
        'type': 'function',
        'label': 'authenticate_user',
        'observations': [],
        'assumptions': []
    }

    graph_data = {
        'nodes': [node_data],
        'edges': []
    }

    score = heuristic.compute_score(
        node_id='func_authenticate',
        graph_data=graph_data,
        visited_nodes=set()
    )

    # Auth-related functions should have high score (lower is better in A*)
    # So high priority = low score
    assert score < 0.5  # High priority


def test_security_heuristic_revisit_penalty():
    """Test that revisiting nodes increases score (lowers priority)."""
    heuristic = SecurityHeuristic()

    node_data = {
        'id': 'func_helper',
        'type': 'function',
        'label': 'helper_function',
        'observations': [],
        'assumptions': []
    }

    graph_data = {
        'nodes': [node_data],
        'edges': []
    }

    # First visit
    score1 = heuristic.compute_score('func_helper', graph_data, set())

    # Second visit (already visited)
    score2 = heuristic.compute_score('func_helper', graph_data, {'func_helper'})

    # Revisit should have higher score (lower priority)
    assert score2 > score1


def test_security_heuristic_information_gain():
    """Test heuristic favors nodes with many unexplored neighbors."""
    heuristic = SecurityHeuristic()

    # Node with many unexplored neighbors
    node1_data = {'id': 'node1', 'type': 'function', 'label': 'func1'}
    node2_data = {'id': 'node2', 'type': 'function', 'label': 'func2'}
    node3_data = {'id': 'node3', 'type': 'function', 'label': 'func3'}

    graph_data = {
        'nodes': [node1_data, node2_data, node3_data],
        'edges': [
            {'source_id': 'node1', 'target_id': 'node2', 'type': 'calls'},
            {'source_id': 'node1', 'target_id': 'node3', 'type': 'calls'}
        ]
    }

    # node1 has 2 unexplored neighbors, should have lower score
    score_node1 = heuristic.compute_score('node1', graph_data, set())

    # node2 has 0 unexplored neighbors (node1 is visited)
    score_node2 = heuristic.compute_score('node2', graph_data, {'node1'})

    assert score_node1 < score_node2  # node1 higher priority


def test_security_heuristic_missing_node():
    """Test heuristic handles missing nodes gracefully."""
    heuristic = SecurityHeuristic()

    graph_data = {
        'nodes': [
            {'id': 'node1', 'type': 'function', 'label': 'func1'}
        ],
        'edges': []
    }

    # Score for non-existent node should return default low priority
    score = heuristic.compute_score('nonexistent', graph_data, set())
    assert score == 1.0  # Default low priority


def test_security_heuristic_isolated_node():
    """Test heuristic handles isolated nodes (no neighbors)."""
    heuristic = SecurityHeuristic()

    isolated_node = {
        'id': 'isolated',
        'type': 'function',
        'label': 'isolated_func'
    }

    graph_data = {
        'nodes': [isolated_node],
        'edges': []  # No edges = isolated node
    }

    score = heuristic.compute_score('isolated', graph_data, set())

    # Should have low priority (no neighbors = no info gain)
    # info_gain = 1.0, risk = 0.7, revisit = 0.0
    # Expected: 0.4*0.7 + 0.4*1.0 + 0.2*0.0 = 0.28 + 0.4 = 0.68
    assert score == pytest.approx(0.68, abs=0.01)


def test_security_heuristic_custom_weights():
    """Test heuristic respects custom weight configuration."""
    # Create heuristic that only considers risk (ignore info gain)
    heuristic_risk_only = SecurityHeuristic(
        risk_weight=1.0,
        info_gain_weight=0.0,
        revisit_penalty=0.0
    )

    high_risk_node = {
        'id': 'auth_node',
        'type': 'function',
        'label': 'authenticate'
    }

    low_risk_node = {
        'id': 'helper',
        'type': 'function',
        'label': 'helper_func'
    }

    graph_data = {
        'nodes': [high_risk_node, low_risk_node],
        'edges': []
    }

    score_high = heuristic_risk_only.compute_score('auth_node', graph_data, set())
    score_low = heuristic_risk_only.compute_score('helper', graph_data, set())

    # With risk_weight=1.0, scores should directly reflect risk scores
    assert score_high < 0.2  # High risk
    assert score_low == 0.7  # Low risk


def test_search_state_frontier_ordering():
    """Test SearchState frontier maintains min-heap ordering."""
    state = SearchState()

    # Add nodes with different f_scores
    node1 = SearchNode('node1', 'graph', g_score=1.0, h_score=2.0, parent=None)  # f=3.0
    node2 = SearchNode('node2', 'graph', g_score=0.5, h_score=1.0, parent=None)  # f=1.5
    node3 = SearchNode('node3', 'graph', g_score=2.0, h_score=0.5, parent=None)  # f=2.5

    state.add_to_frontier(node1)
    state.add_to_frontier(node2)
    state.add_to_frontier(node3)

    # Pop should return nodes in ascending f_score order
    first = state.pop_frontier()
    second = state.pop_frontier()
    third = state.pop_frontier()

    assert first.node_id == 'node2'  # f=1.5
    assert second.node_id == 'node3'  # f=2.5
    assert third.node_id == 'node1'  # f=3.0


def test_search_state_explored_tracking():
    """Test SearchState correctly tracks explored nodes."""
    state = SearchState()

    assert not state.is_explored('node1')

    state.mark_explored('node1')
    assert state.is_explored('node1')
    assert not state.is_explored('node2')

    state.mark_explored('node2')
    assert state.is_explored('node1')
    assert state.is_explored('node2')

    # Verify explored set contains both
    assert 'node1' in state.explored
    assert 'node2' in state.explored
    assert len(state.explored) == 2


def test_astar_search_finds_path():
    """Test A* finds optimal path in simple graph."""
    search = AStarSearch()

    # Simple graph: A -> B -> C
    #                \-> D -> C
    graph_data = {
        'nodes': [
            {'id': 'A', 'type': 'function', 'label': 'start'},
            {'id': 'B', 'type': 'function', 'label': 'middle1'},
            {'id': 'C', 'type': 'function', 'label': 'goal_authenticate'},  # High priority
            {'id': 'D', 'type': 'function', 'label': 'middle2'}
        ],
        'edges': [
            {'source_id': 'A', 'target_id': 'B', 'type': 'calls'},
            {'source_id': 'B', 'target_id': 'C', 'type': 'calls'},
            {'source_id': 'A', 'target_id': 'D', 'type': 'calls'},
            {'source_id': 'D', 'target_id': 'C', 'type': 'calls'}
        ]
    }

    # Search for high-priority authentication node
    path = search.search(
        start_node='A',
        graph_data=graph_data,
        max_steps=10
    )

    # Should find C (authentication node)
    assert path is not None
    assert 'C' in path
    assert path[0] == 'A'  # Start
    assert path[-1] == 'C'  # Goal (auth node)


def test_astar_search_prioritizes_security():
    """Test A* prioritizes security-critical nodes."""
    search = AStarSearch()

    # Graph with security-critical path
    graph_data = {
        'nodes': [
            {'id': 'start', 'type': 'function', 'label': 'main'},
            {'id': 'helper', 'type': 'function', 'label': 'helper'},
            {'id': 'auth', 'type': 'function', 'label': 'authenticate'},
            {'id': 'normal', 'type': 'function', 'label': 'process'}
        ],
        'edges': [
            {'source_id': 'start', 'target_id': 'helper', 'type': 'calls'},
            {'source_id': 'start', 'target_id': 'auth', 'type': 'calls'},
            {'source_id': 'start', 'target_id': 'normal', 'type': 'calls'}
        ]
    }

    # Get next best node to explore
    path = search.search(start_node='start', graph_data=graph_data, max_steps=5)

    # Should include auth node early due to high security priority
    assert 'auth' in path[:3]  # In first 3 nodes


def test_astar_search_max_steps():
    """Test A* respects max_steps limit."""
    search = AStarSearch()

    # Large graph
    nodes = [{'id': f'node{i}', 'type': 'function', 'label': f'func{i}'}
             for i in range(100)]
    edges = [{'source_id': f'node{i}', 'target_id': f'node{i+1}', 'type': 'calls'}
             for i in range(99)]

    graph_data = {'nodes': nodes, 'edges': edges}

    # Search with limit
    path = search.search(start_node='node0', graph_data=graph_data, max_steps=10)

    # Should stop after max_steps
    assert len(path) <= 10
