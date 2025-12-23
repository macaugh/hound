"""A* search algorithm with security-aware heuristics."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class SearchNode:
    """Node in A* search space."""

    # Priority queue ordering field (must be first for dataclass ordering)
    f_score: float = field(init=False, repr=False)

    # Actual node data
    node_id: str = field(compare=False)
    graph_name: str = field(compare=False)
    g_score: float = field(compare=False)  # Cost from start
    h_score: float = field(compare=False)  # Heuristic to goal
    parent: SearchNode | None = field(default=None, compare=False)

    def __post_init__(self):
        self.f_score = self.g_score + self.h_score


class SearchState:
    """Maintains A* search state."""

    def __init__(self):
        self.explored: set[str] = set()  # Explored node IDs
        self.frontier: list[SearchNode] = []  # Priority queue (heap)
        self.node_scores: dict[str, float] = {}  # Best g_score per node

    def add_to_frontier(self, node: SearchNode):
        """Add node to frontier (priority queue)."""
        heapq.heappush(self.frontier, node)
        self.node_scores[node.node_id] = node.g_score

    def pop_frontier(self) -> SearchNode | None:
        """Pop best node from frontier."""
        if not self.frontier:
            return None
        return heapq.heappop(self.frontier)

    def mark_explored(self, node_id: str):
        """Mark node as explored."""
        self.explored.add(node_id)

    def is_explored(self, node_id: str) -> bool:
        """Check if node was explored."""
        return node_id in self.explored


class SecurityHeuristic:
    """Security-aware heuristic for A* search.

    Combines:
    1. Security risk score (vulnerability patterns)
    2. Information gain (unexplored neighbors)
    3. Revisit penalty (avoid redundant exploration)

    Lower score = higher priority for exploration.
    """

    # High-risk patterns (CVE-style indicators)
    HIGH_RISK_PATTERNS = [
        'auth', 'authenticate', 'login', 'password', 'token',
        'validate', 'verify', 'check', 'permission', 'access',
        'transfer', 'withdraw', 'mint', 'burn', 'approve',
        'execute', 'call', 'delegatecall', 'send', 'eval'
    ]

    # Medium-risk patterns
    MEDIUM_RISK_PATTERNS = [
        'init', 'initialize', 'constructor', 'deploy',
        'update', 'set', 'modify', 'change',
        'balance', 'amount', 'value', 'price'
    ]

    def __init__(self,
                 risk_weight: float = 0.4,
                 info_gain_weight: float = 0.4,
                 revisit_penalty: float = 0.2):
        """Initialize heuristic with weights."""
        self.risk_weight = risk_weight
        self.info_gain_weight = info_gain_weight
        self.revisit_penalty = revisit_penalty

        # Performance caches
        self._node_index: dict[str, dict[str, Any]] = {}
        self._adjacency_list: dict[str, set[str]] = {}
        self._graph_hash: int | None = None

    def _compute_graph_hash(self, graph_data: dict[str, Any]) -> int:
        """Compute hash of graph structure for cache invalidation."""
        # Hash based on node IDs and edge structure
        node_ids = tuple(sorted(n.get('id', '') for n in graph_data.get('nodes', [])))
        edge_tuples = tuple(sorted(
            (e.get('source_id', ''), e.get('target_id', ''))
            for e in graph_data.get('edges', [])
        ))
        return hash((node_ids, edge_tuples))

    def _rebuild_caches(self, graph_data: dict[str, Any]):
        """Rebuild node index and adjacency list caches."""
        # Build node index: O(N)
        self._node_index = {
            node['id']: node
            for node in graph_data.get('nodes', [])
            if 'id' in node
        }

        # Build adjacency list: O(E)
        self._adjacency_list = {}
        for edge in graph_data.get('edges', []):
            source_id = edge.get('source_id')
            target_id = edge.get('target_id')

            if not source_id or not target_id:
                continue

            # Exclude self-loops
            if source_id != target_id:
                self._adjacency_list.setdefault(source_id, set()).add(target_id)
                self._adjacency_list.setdefault(target_id, set()).add(source_id)

    def compute_score(self,
                     node_id: str,
                     graph_data: dict[str, Any],
                     visited_nodes: set[str]) -> float:
        """Compute heuristic score for a node.

        Args:
            node_id: Node to score
            graph_data: Graph containing the node
            visited_nodes: Set of already-visited node IDs

        Returns:
            Heuristic score (lower = higher priority)
        """
        # Input validation
        if not isinstance(graph_data, dict):
            raise ValueError("graph_data must be a dictionary")

        # Convert visited_nodes to set if needed (performance optimization)
        if not isinstance(visited_nodes, set):
            visited_nodes = set(visited_nodes)

        # Rebuild caches if graph changed
        current_hash = self._compute_graph_hash(graph_data)
        if current_hash != self._graph_hash:
            self._rebuild_caches(graph_data)
            self._graph_hash = current_hash

        # Find node in graph (now uses cache)
        node = self._find_node(node_id)
        if not node:
            return 1.0  # Default low priority for missing nodes

        # Component 1: Security risk score (0.0 = high risk, 1.0 = low risk)
        risk_score = self._compute_risk_score(node)

        # Component 2: Information gain (0.0 = many unexplored neighbors, 1.0 = none)
        info_gain = self._compute_information_gain(node_id, visited_nodes)

        # Component 3: Revisit penalty (0.0 = not visited, 1.0 = already visited)
        revisit = 1.0 if node_id in visited_nodes else 0.0

        # Combine weighted scores
        total_score = (
            self.risk_weight * risk_score +
            self.info_gain_weight * info_gain +
            self.revisit_penalty * revisit
        )

        return total_score

    def _find_node(self, node_id: str) -> dict | None:
        """Find node by ID using cached index.

        Returns:
            Node dict if found, None otherwise
        """
        return self._node_index.get(node_id)

    def _compute_risk_score(self, node: dict[str, Any]) -> float:
        """Compute security risk score for node.

        Returns:
            0.0 = very high risk (should explore first)
            1.0 = low risk (can explore later)

        Uses continuous scoring based on pattern match count for smoother
        priority gradients and consistent weighting with other components.
        """
        label = node.get('label', '').lower()
        node_type = node.get('type', '').lower()

        # Count pattern matches (more matches = higher risk)
        high_risk_matches = sum(
            1 for pattern in self.HIGH_RISK_PATTERNS
            if pattern in label or pattern in node_type
        )
        medium_risk_matches = sum(
            1 for pattern in self.MEDIUM_RISK_PATTERNS
            if pattern in label or pattern in node_type
        )

        if high_risk_matches > 0:
            # Scale: 1 match = 0.1, 2+ matches approach 0.0 (very high priority)
            return max(0.0, 0.2 - 0.1 * high_risk_matches)
        elif medium_risk_matches > 0:
            # Scale: 1 match = 0.4, 2+ matches approach 0.3
            return max(0.3, 0.5 - 0.1 * medium_risk_matches)
        else:
            # Default: low risk
            return 0.7

    def _compute_information_gain(self,
                                  node_id: str,
                                  visited_nodes: set[str]) -> float:
        """Compute information gain score using cached adjacency list.

        Returns:
            0.0 = many unexplored neighbors (high info gain)
            1.0 = no unexplored neighbors (low info gain)
        """
        # Get neighbors from cached adjacency list (O(1) lookup)
        neighbors = self._adjacency_list.get(node_id, set())

        if not neighbors:
            return 1.0  # No neighbors = no info gain

        # Count unexplored neighbors
        unexplored = neighbors - visited_nodes
        unexplored_ratio = len(unexplored) / len(neighbors)

        # Invert: more unexplored = lower score (higher priority)
        return 1.0 - unexplored_ratio


class AStarSearch:
    """A* search algorithm for intelligent code exploration."""

    def __init__(self, heuristic: SecurityHeuristic | None = None):
        """Initialize A* search."""
        self.heuristic = heuristic or SecurityHeuristic()
        self.state = SearchState()

    def search(self,
               start_node: str,
               graph_data: dict[str, Any],
               max_steps: int = 20) -> list[str]:
        """Execute A* search from start node.

        Args:
            start_node: Starting node ID
            graph_data: Graph containing nodes and edges
            max_steps: Maximum search steps (prevents infinite loops)

        Returns:
            List of node IDs in exploration order
        """
        # Reset state
        self.state = SearchState()

        # Initialize with start node
        start = SearchNode(
            node_id=start_node,
            graph_name="unknown",
            g_score=0.0,
            h_score=self.heuristic.compute_score(start_node, graph_data, set()),
            parent=None
        )
        self.state.add_to_frontier(start)

        path = []
        steps = 0

        while steps < max_steps:
            # Get best node from frontier
            current = self.state.pop_frontier()
            if current is None:
                break  # No more nodes to explore

            # Skip if already explored
            if self.state.is_explored(current.node_id):
                continue

            # Add to path and mark explored
            path.append(current.node_id)
            self.state.mark_explored(current.node_id)
            steps += 1

            # Expand neighbors
            neighbors = self._get_neighbors(current.node_id, graph_data)
            for neighbor_id in neighbors:
                if self.state.is_explored(neighbor_id):
                    continue

                # Compute scores
                g_score = current.g_score + 1.0  # Uniform edge cost
                h_score = self.heuristic.compute_score(
                    neighbor_id,
                    graph_data,
                    self.state.explored
                )

                # Add to frontier if better path or not seen
                if (neighbor_id not in self.state.node_scores or
                    g_score < self.state.node_scores[neighbor_id]):

                    neighbor_node = SearchNode(
                        node_id=neighbor_id,
                        graph_name="unknown",
                        g_score=g_score,
                        h_score=h_score,
                        parent=current
                    )
                    self.state.add_to_frontier(neighbor_node)

        return path

    def _get_neighbors(self, node_id: str, graph_data: dict[str, Any]) -> list[str]:
        """Get neighbor node IDs for a given node."""
        neighbors = []
        for edge in graph_data.get('edges', []):
            if edge.get('source_id') == node_id:
                neighbors.append(edge.get('target_id'))
            if edge.get('target_id') == node_id:
                neighbors.append(edge.get('source_id'))
        return neighbors
