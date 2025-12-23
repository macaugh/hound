"""Benchmark A* search vs baseline exploration."""
import time
import random
from analysis.search.astar import AStarSearch


def generate_test_graph(num_nodes: int, num_edges: int, num_vuln: int):
    """Generate random graph with vulnerabilities."""
    nodes = []
    for i in range(num_nodes):
        label = 'authenticate' if i < num_vuln else f'func{i}'
        nodes.append({
            'id': f'node{i}',
            'type': 'function',
            'label': label
        })

    edges = []
    for _ in range(num_edges):
        src = random.randint(0, num_nodes-1)
        dst = random.randint(0, num_nodes-1)
        if src != dst:
            edges.append({
                'source_id': f'node{src}',
                'target_id': f'node{dst}',
                'type': 'calls'
            })

    return {'nodes': nodes, 'edges': edges}


def benchmark_astar(graph_data, num_runs=10):
    """Benchmark A* search."""
    search = AStarSearch()
    times = []

    for _ in range(num_runs):
        start = time.time()
        path = search.search('node0', graph_data, max_steps=50)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'avg_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'path_length': len(path) if path else 0
    }


def main():
    """Run benchmarks."""
    print("A* Search Benchmark")
    print("=" * 60)

    test_cases = [
        (50, 100, 5),    # Small
        (100, 300, 10),  # Medium
        (500, 1500, 20), # Large
    ]

    for num_nodes, num_edges, num_vuln in test_cases:
        print(f"\nGraph: {num_nodes} nodes, {num_edges} edges, {num_vuln} vulnerabilities")

        graph = generate_test_graph(num_nodes, num_edges, num_vuln)
        results = benchmark_astar(graph, num_runs=10)

        print(f"  Avg time: {results['avg_time']*1000:.2f}ms")
        print(f"  Path length: {results['path_length']}")
        if results['avg_time'] > 0 and results['path_length'] > 0:
            print(f"  Nodes/sec: {results['path_length'] / results['avg_time']:.0f}")


if __name__ == '__main__':
    main()
