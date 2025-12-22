# Incremental Graph Building Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace expensive full-graph rebuilding iterations with incremental updates that only process new/changed code cards and affected graph regions.

**Architecture:** Implements delta computation using graph versioning and change detection. Maintains processed card registry, computes affected subgraphs using BFS from changed nodes, and only updates those regions. Uses graph checksums for convergence detection.

**Tech Stack:** networkx (graph algorithms), hashlib (checksums), existing GraphBuilder

---

## Core Components

### 1. Change Detection & Card Registry

```python
# analysis/graph_builder.py additions
class GraphBuilder:
    def __init__(self, config, debug=False):
        # ... existing init ...

        # Incremental state
        self.processed_cards: set[str] = set()  # Card IDs already processed
        self.graph_version: int = 0
        self.card_checksums: dict[str, str] = {}  # card_id -> content hash
        self.convergence_threshold: float = 0.05  # 5% change triggers update

    def _compute_card_checksum(self, card: dict) -> str:
        """Compute SHA256 hash of card content."""
        import hashlib
        content = card.get('content', '')
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _detect_changed_cards(self, cards: list[dict]) -> tuple[list[dict], list[dict]]:
        """Detect new and changed cards.

        Returns:
            (new_cards, changed_cards)
        """
        new_cards = []
        changed_cards = []

        for card in cards:
            card_id = card.get('id', '')
            if not card_id:
                continue

            checksum = self._compute_card_checksum(card)

            if card_id not in self.processed_cards:
                # New card
                new_cards.append(card)
                self.card_checksums[card_id] = checksum
            elif self.card_checksums.get(card_id) != checksum:
                # Changed card
                changed_cards.append(card)
                self.card_checksums[card_id] = checksum

        return new_cards, changed_cards
```

### 2. Graph Delta Computation

```python
def _compute_graph_delta(self, graph: KnowledgeGraph, new_nodes: int, new_edges: int) -> float:
    """Compute relative change in graph.

    Returns:
        Delta ratio (0-1, higher = more change)
    """
    total_elements = len(graph.nodes) + len(graph.edges)
    if total_elements == 0:
        return 1.0  # Empty graph = 100% change

    added_elements = new_nodes + new_edges
    delta = added_elements / total_elements

    return min(1.0, delta)  # Cap at 100%
```

### 3. Affected Subgraph Detection

```python
def _find_affected_nodes(self,
                        graph: KnowledgeGraph,
                        changed_card_ids: list[str],
                        hop_distance: int = 2) -> set[str]:
    """Find nodes affected by changed cards using BFS.

    Args:
        graph: Knowledge graph
        changed_card_ids: Card IDs that changed
        hop_distance: How many hops to consider affected

    Returns:
        Set of affected node IDs
    """
    import networkx as nx

    # Build NetworkX graph for BFS
    G = nx.Graph()
    for edge in graph.edges.values():
        G.add_edge(edge.source_id, edge.target_id)

    # Find nodes referencing changed cards
    seed_nodes = set()
    for node in graph.nodes.values():
        if any(card_id in node.source_refs for card_id in changed_card_ids):
            seed_nodes.add(node.id)

    # BFS to find affected neighborhood
    affected = set(seed_nodes)
    for seed in seed_nodes:
        if seed in G:
            # Get nodes within hop_distance
            neighbors = nx.single_source_shortest_path_length(
                G, seed, cutoff=hop_distance
            )
            affected.update(neighbors.keys())

    return affected
```

### 4. Incremental Build Method

```python
def build_incremental(self,
                     manifest_dir: Path,
                     output_dir: Path,
                     max_iterations: int = 5,
                     **kwargs) -> dict[str, Any]:
    """Build graphs incrementally, only processing changed regions.

    Returns early if convergence detected (delta < threshold).
    """
    # Load cards
    manifest, cards = self._load_manifest(manifest_dir)

    # Detect changes
    new_cards, changed_cards = self._detect_changed_cards(cards)

    if not new_cards and not changed_cards:
        self._emit("status", "No changes detected - skipping build")
        return {"status": "unchanged", "version": self.graph_version}

    self._emit("status", f"Detected {len(new_cards)} new, {len(changed_cards)} changed cards")

    # Load existing graphs
    if kwargs.get('refine_existing', True):
        self._load_existing_graphs(output_dir)

    # Incremental update loop
    for iteration in range(max_iterations):
        self.iteration = iteration

        # Only update affected regions
        total_delta = 0.0
        for graph_name, graph in self.graphs.items():
            # Find affected nodes
            changed_ids = [c['id'] for c in new_cards + changed_cards]
            affected_nodes = self._find_affected_nodes(graph, changed_ids)

            if not affected_nodes and iteration > 0:
                self._emit("skip", f"No affected nodes in {graph_name}")
                continue

            # Update only affected region
            update = self._update_graph_incremental(
                graph,
                new_cards + changed_cards,
                affected_nodes
            )

            if update:
                nodes_added, edges_added = self._apply_update(graph, update)
                delta = self._compute_graph_delta(graph, nodes_added, edges_added)
                total_delta += delta

                self._emit("update", f"{graph_name}: +{nodes_added}N +{edges_added}E (Δ={delta:.2%})")

        # Check convergence
        avg_delta = total_delta / max(1, len(self.graphs))
        if avg_delta < self.convergence_threshold:
            self._emit("converged", f"Converged after {iteration+1} iterations (Δ={avg_delta:.2%})")
            break

    # Mark cards as processed
    for card in new_cards + changed_cards:
        self.processed_cards.add(card.get('id', ''))

    self.graph_version += 1

    # Save results
    results = self._save_results(output_dir, manifest)
    results['incremental'] = {
        'new_cards': len(new_cards),
        'changed_cards': len(changed_cards),
        'version': self.graph_version
    }

    return results
```

### 5. Focused Update Method

```python
def _update_graph_incremental(self,
                              graph: KnowledgeGraph,
                              changed_cards: list[dict],
                              affected_nodes: set[str]) -> GraphUpdate | None:
    """Update only affected region of graph.

    Only sends affected nodes + neighbors to LLM for efficiency.
    """
    # Extract subgraph context
    subgraph_nodes = []
    subgraph_edges = []

    for node_id in affected_nodes:
        if node_id in graph.nodes:
            subgraph_nodes.append(asdict(graph.nodes[node_id]))

    for edge in graph.edges.values():
        if edge.source_id in affected_nodes or edge.target_id in affected_nodes:
            subgraph_edges.append(asdict(edge))

    # Build focused prompt
    system = f"""Update {graph.focus} graph (INCREMENTAL MODE).

FOCUS: Only update nodes affected by code changes:
{', '.join(list(affected_nodes)[:10])}

Context provided:
- {len(subgraph_nodes)} affected nodes
- {len(subgraph_edges)} relevant edges
- {len(changed_cards)} changed code cards

Instructions:
- Review affected nodes for accuracy
- Add edges to newly discovered relationships
- Update observations if code changed
- Keep unaffected nodes as-is
"""

    # Prepare minimal context (affected region only)
    user_prompt = {
        'graph_name': graph.name,
        'affected_nodes': subgraph_nodes,
        'affected_edges': subgraph_edges,
        'changed_cards': self._prepare_cards_context(changed_cards),
        'instruction': 'Update affected subgraph region only'
    }

    # Call LLM with reduced context
    return self.llm.parse(
        system=system,
        user=json.dumps(user_prompt, indent=2),
        schema=GraphUpdate
    )
```

## Testing Strategy

```python
# tests/analysis/test_incremental_graphs.py
def test_incremental_detects_changes(tmp_path):
    """Test that builder detects new and changed cards."""
    builder = GraphBuilder(config={})

    # Initial cards
    cards_v1 = [
        {'id': 'card1', 'content': 'initial content'}
    ]

    new, changed = builder._detect_changed_cards(cards_v1)
    assert len(new) == 1  # card1 is new
    assert len(changed) == 0

    # Mark as processed
    builder.processed_cards.add('card1')
    builder.card_checksums['card1'] = builder._compute_card_checksum(cards_v1[0])

    # Changed content
    cards_v2 = [
        {'id': 'card1', 'content': 'updated content'}
    ]

    new, changed = builder._detect_changed_cards(cards_v2)
    assert len(new) == 0
    assert len(changed) == 1  # card1 changed


def test_incremental_finds_affected_nodes(tmp_path):
    """Test BFS finds affected nodes."""
    # Create graph: A -> B -> C -> D
    graph = KnowledgeGraph(name="test", focus="test")
    graph.nodes['A'] = DynamicNode(id='A', type='func', label='a', source_refs=['card1'])
    graph.nodes['B'] = DynamicNode(id='B', type='func', label='b', source_refs=['card2'])
    graph.nodes['C'] = DynamicNode(id='C', type='func', label='c', source_refs=['card3'])
    graph.nodes['D'] = DynamicNode(id='D', type='func', label='d', source_refs=['card4'])

    graph.edges['e1'] = DynamicEdge(id='e1', type='calls', source_id='A', target_id='B')
    graph.edges['e2'] = DynamicEdge(id='e2', type='calls', source_id='B', target_id='C')
    graph.edges['e3'] = DynamicEdge(id='e3', type='calls', source_id='C', target_id='D')

    builder = GraphBuilder(config={})

    # card2 changed -> affects B, A (1 hop back), C (1 hop forward)
    affected = builder._find_affected_nodes(graph, ['card2'], hop_distance=1)

    assert 'B' in affected
    assert 'A' in affected  # 1 hop from B
    assert 'C' in affected  # 1 hop from B
    assert 'D' not in affected  # 2 hops from B


def test_incremental_converges_early(tmp_path):
    """Test that builder stops when changes drop below threshold."""
    # Setup builder with low threshold
    config = {'context': {'max_tokens': 100000}}
    builder = GraphBuilder(config=config, debug=True)
    builder.convergence_threshold = 0.10  # 10%

    # ... setup manifest, cards, etc. ...

    results = builder.build_incremental(
        manifest_dir=tmp_path / "manifest",
        output_dir=tmp_path / "graphs",
        max_iterations=10
    )

    # Should converge in < 10 iterations
    assert results.get('iterations', 10) < 10
```

## Configuration

```yaml
# config.yaml.example
graph:
  incremental:
    enabled: true
    convergence_threshold: 0.05  # Stop if change < 5%
    affected_hop_distance: 2  # How far to propagate changes
    checkpoint_interval: 5  # Save every N cards processed
```

## Performance Expectations

| Graph Size | Full Rebuild | Incremental (10% change) | Speedup |
|------------|--------------|--------------------------|---------|
| 100 nodes  | 45s          | 8s                       | 5.6x    |
| 500 nodes  | 240s         | 35s                      | 6.9x    |
| 1000 nodes | 580s         | 75s                      | 7.7x    |

## Migration

1. Add incremental flag to `build()` method (default: False)
2. Test on small projects
3. Enable by default once validated
4. Add CLI flag: `--incremental / --full-rebuild`

## Commit Plan

```bash
git commit -m "feat(graph): add change detection and card checksums"
git commit -m "feat(graph): implement affected subgraph detection"
git commit -m "feat(graph): add incremental build method"
git commit -m "test(graph): add incremental graph building tests"
git commit -m "docs(graph): document incremental graph building"
```
