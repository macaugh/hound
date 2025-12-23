"""Benchmark semantic duplicate detection accuracy and performance."""
import time
from analysis.hypothesis.semantic_matcher import SemanticMatcher, is_duplicate_hypothesis


# Test cases: (new_hyp, existing_hyps, should_be_duplicate)
TEST_CASES = [
    # True positives (should detect duplicates)
    (
        {'title': 'Missing auth in transfer', 'node_refs': ['func_transfer']},
        [{'title': 'Transfer lacks authentication', 'node_refs': ['func_transfer']}],
        True
    ),
    (
        {'title': 'Reentrancy vulnerability in withdraw', 'node_refs': ['func_withdraw']},
        [{'title': 'Withdraw function has reentrancy bug', 'node_refs': ['func_withdraw']}],
        True
    ),
    (
        {'title': 'Integer overflow in balance calculation', 'node_refs': ['state_balance']},
        [{'title': 'Balance calc has integer overflow', 'node_refs': ['state_balance']}],
        True
    ),
    (
        {'title': 'Access control missing in admin function', 'node_refs': ['func_admin']},
        [{'title': 'Admin function lacks access control', 'node_refs': ['func_admin']}],
        True
    ),

    # True negatives (should allow - different vulnerabilities)
    (
        {'title': 'Integer overflow in balance', 'node_refs': ['state_balance']},
        [{'title': 'Missing auth in transfer', 'node_refs': ['func_transfer']}],
        False
    ),
    (
        {'title': 'Reentrancy in withdraw', 'node_refs': ['func_withdraw']},
        [{'title': 'Integer overflow in deposit', 'node_refs': ['func_deposit']}],
        False
    ),
    (
        {'title': 'SQL injection in query', 'node_refs': ['func_query']},
        [{'title': 'XSS vulnerability in render', 'node_refs': ['func_render']}],
        False
    ),

    # True negatives (similar text, different nodes - should allow)
    (
        {'title': 'Missing auth check', 'node_refs': ['func_transfer']},
        [{'title': 'Missing auth check', 'node_refs': ['func_withdraw']}],
        False
    ),
    (
        {'title': 'Access control bypass', 'node_refs': ['func_admin']},
        [{'title': 'Authorization bypass', 'node_refs': ['func_user']}],
        False
    ),

    # Edge cases - synonyms (should detect with high similarity + node overlap)
    (
        {'title': 'Missing authentication check', 'node_refs': ['func_login']},
        [{'title': 'Authentication check missing', 'node_refs': ['func_login']}],
        True
    ),
    (
        {'title': 'Buffer overflow vulnerability', 'node_refs': ['func_copy']},
        [{'title': 'Buffer overflow in copy operation', 'node_refs': ['func_copy']}],
        True
    ),
]


def evaluate_accuracy():
    """Evaluate accuracy of semantic matching."""
    matcher = SemanticMatcher(threshold=0.85)

    correct = 0
    total = len(TEST_CASES)

    results = []

    for new_hyp, existing, expected_dup in TEST_CASES:
        is_dup, _ = is_duplicate_hypothesis(new_hyp, existing, matcher)

        is_correct = (is_dup == expected_dup)
        if is_correct:
            correct += 1

        results.append({
            'new': new_hyp['title'],
            'existing': existing[0]['title'],
            'expected': expected_dup,
            'got': is_dup,
            'correct': is_correct
        })

    accuracy = correct / total

    return accuracy, results


def benchmark_performance():
    """Benchmark matching performance."""
    matcher = SemanticMatcher()

    # Generate test data
    test_titles = [
        f"Vulnerability {i} in function_name_{i}"
        for i in range(100)
    ]

    # Warm up (load model)
    matcher.compute_similarity(test_titles[0], test_titles[1])

    # Benchmark similarity computation
    times = []
    for i in range(100):
        start = time.time()
        matcher.compute_similarity(test_titles[i], test_titles[(i+1) % 100])
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        'avg_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'throughput': 1 / avg_time
    }


def benchmark_cache_performance():
    """Benchmark cache hit performance."""
    matcher = SemanticMatcher()

    text1 = "Authentication bypass vulnerability"
    text2 = "Authorization issue in handler"

    # First call - no cache
    start = time.time()
    matcher.compute_similarity(text1, text2)
    first_time = time.time() - start

    # Second call - should use cache
    start = time.time()
    matcher.compute_similarity(text1, text2)
    cached_time = time.time() - start

    speedup = first_time / cached_time if cached_time > 0 else 0

    return {
        'first_call_ms': first_time * 1000,
        'cached_call_ms': cached_time * 1000,
        'speedup': speedup
    }


def main():
    """Run benchmarks."""
    print("=" * 70)
    print("Semantic Duplicate Detection Benchmark")
    print("=" * 70)

    print("\n1. Accuracy Evaluation")
    print("-" * 70)
    accuracy, results = evaluate_accuracy()
    print(f"   Overall Accuracy: {accuracy*100:.1f}%")
    print(f"   Correct: {sum(r['correct'] for r in results)}/{len(results)}")

    # Count true positives, false positives, etc.
    tp = sum(1 for r in results if r['expected'] and r['got'] and r['correct'])
    fp = sum(1 for r in results if not r['expected'] and r['got'])
    tn = sum(1 for r in results if not r['expected'] and not r['got'] and r['correct'])
    fn = sum(1 for r in results if r['expected'] and not r['got'])

    print(f"\n   Confusion Matrix:")
    print(f"   - True Positives:  {tp}")
    print(f"   - False Positives: {fp}")
    print(f"   - True Negatives:  {tn}")
    print(f"   - False Negatives: {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"   - Precision: {precision*100:.1f}%")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"   - Recall: {recall*100:.1f}%")

    print(f"\n   Detailed Results:")
    for i, r in enumerate(results, 1):
        status = "✓" if r['correct'] else "✗"
        exp_str = "DUP" if r['expected'] else "ALLOW"
        got_str = "DUP" if r['got'] else "ALLOW"
        print(f"   {status} Case {i:2d}: Expected {exp_str:5s}, Got {got_str:5s}")
        print(f"      New:      {r['new']}")
        print(f"      Existing: {r['existing']}")

    print("\n2. Performance Benchmark")
    print("-" * 70)
    perf = benchmark_performance()
    print(f"   Average time:  {perf['avg_time_ms']:.2f}ms per comparison")
    print(f"   Min time:      {perf['min_time_ms']:.2f}ms")
    print(f"   Max time:      {perf['max_time_ms']:.2f}ms")
    print(f"   Throughput:    {perf['throughput']:.0f} comparisons/sec")

    print("\n3. Cache Performance")
    print("-" * 70)
    cache_perf = benchmark_cache_performance()
    print(f"   First call:    {cache_perf['first_call_ms']:.2f}ms")
    print(f"   Cached call:   {cache_perf['cached_call_ms']:.2f}ms")
    print(f"   Speedup:       {cache_perf['speedup']:.1f}x faster")

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)

    # Summary
    if accuracy >= 0.90:
        print(f"\n✓ Accuracy {accuracy*100:.1f}% meets target (≥90%)")
    else:
        print(f"\n✗ Accuracy {accuracy*100:.1f}% below target (≥90%)")

    if perf['avg_time_ms'] <= 10:
        print(f"✓ Performance {perf['avg_time_ms']:.2f}ms meets target (≤10ms)")
    else:
        print(f"✗ Performance {perf['avg_time_ms']:.2f}ms exceeds target (≤10ms)")


if __name__ == '__main__':
    main()
