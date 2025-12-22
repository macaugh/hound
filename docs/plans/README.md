# Hound Algorithmic Improvements - Implementation Plans

This directory contains comprehensive implementation plans for 5 major algorithmic improvements to the Hound security audit tool, based on latest research in algorithms, mathematics, and information theory.

## Overview

Each improvement has:
1. **Dedicated git worktree** for isolated development and A/B testing
2. **Detailed implementation plan** with TDD approach and bite-sized tasks
3. **Test suite** for validation
4. **Benchmarks** for performance measurement

## Improvements

### 1. A* Search with Security-Aware Heuristics
**File**: `2025-12-22-astar-search.md`
**Worktree**: `../hound-astar-search`
**Branch**: `feature/astar-search`

**Problem**: Agent uses ad-hoc exploration (random walk), leading to O(n!) worst-case complexity and missing vulnerabilities.

**Solution**: Implement A* search with heuristic combining:
- Security risk scoring (CVE patterns, auth, crypto)
- Information gain (unexplored neighbors, centrality)
- Revisit penalties (avoid redundancy)

**Expected Impact**:
- 60-70% faster to find high-priority vulnerabilities
- 80% reduction in redundant node visits
- Provable optimality guarantees

---

### 2. Semantic Duplicate Detection
**File**: `2025-12-22-semantic-dedup.md`
**Worktree**: `../hound-semantic-dedup`
**Branch**: `feature/semantic-dedup`

**Problem**: Only exact title matching for duplicate detection, missing semantic duplicates with different wording.

**Solution**: Sentence transformers + cosine similarity + node overlap:
- Use `all-MiniLM-L6-v2` for embeddings
- Similarity threshold: 0.85 (configurable)
- Node overlap (Jaccard): 0.3 minimum
- Both conditions must be met

**Expected Impact**:
- 90%+ duplicate detection accuracy
- Reduces duplicate hypotheses by 60-70%
- <10ms per comparison

---

### 3. Bayesian Confidence Scoring
**File**: `2025-12-22-bayesian-confidence.md`
**Worktree**: `../hound-bayesian-confidence`
**Branch**: `feature/bayesian-confidence`

**Problem**: Arbitrary confidence mappings ("high"=0.9, "medium"=0.6) with no uncertainty quantification.

**Solution**: Beta-Binomial conjugate priors:
- Maintain alpha/beta parameters
- Update with Bayes' Rule as evidence accumulates
- Provide credible intervals (90%, 95%)
- Proper probability distributions

**Expected Impact**:
- Proper uncertainty quantification
- Scientifically defensible confidence claims
- Better decision-making on investigation prioritization

---

### 4. Incremental Graph Building
**File**: `2025-12-22-incremental-graphs.md`
**Worktree**: `../hound-incremental-graphs`
**Branch**: `feature/incremental-graphs`

**Problem**: Full graph rebuilds on every iteration waste compute (O(iterations × cards × nodes)).

**Solution**: Incremental updates with delta computation:
- Track processed cards with checksums
- Detect changed/new cards
- Compute affected subgraphs (BFS from changed nodes)
- Update only affected regions
- Early stopping on convergence (< 5% change)

**Expected Impact**:
- 5-7x speedup on subsequent builds
- Enables real-time graph updates
- Reduces LLM API costs by 60-80%

---

### 5. Attention-Based Context Management
**File**: `2025-12-22-attention-context.md`
**Worktree**: `../hound-attention-context`
**Branch**: `feature/attention-context`

**Problem**: All context treated equally; token budget wasted on irrelevant information.

**Solution**: Transformer-style attention mechanism:
- Compute attention scores (query vs context similarity)
- Include high-attention parts + critical sections
- Adaptive compression for low-attention parts
- Softmax normalization

**Expected Impact**:
- 30-60% token reduction
- Better LLM focus on relevant info
- Fits larger investigations in context

---

## Worktree Structure

```
active/
├── hound/                       # Main development (main branch)
├── hound-astar-search/          # A* search implementation
├── hound-semantic-dedup/        # Semantic duplicate detection
├── hound-bayesian-confidence/   # Bayesian confidence scoring
├── hound-incremental-graphs/    # Incremental graph building
└── hound-attention-context/     # Attention-based context
```

## Execution Guide

### Option 1: Execute Single Plan (Recommended)

Navigate to worktree and use the `executing-plans` skill:

```bash
# Example: A* search
cd ../hound-astar-search

# Start new Claude session in this directory
# Then say: "Use superpowers:executing-plans to implement docs/plans/2025-12-22-astar-search.md"
```

The executing-plans skill will:
- Work through tasks in batches
- Stop for review after each batch
- Allow you to test incrementally
- Create proper commits

### Option 2: Subagent-Driven Development

Stay in main hound directory and dispatch subagents:

```bash
# In main session, say:
"Use superpowers:subagent-driven-development to implement the A* search plan"
```

Each task gets a fresh subagent + code review between tasks.

### Option 3: Manual Implementation

1. Check out the worktree:
   ```bash
   cd ../hound-astar-search
   ```

2. Follow the plan step-by-step:
   ```bash
   # Open plan
   cat docs/plans/2025-12-22-astar-search.md

   # Follow Task 1, Step 1, 2, 3, etc.
   ```

3. Run tests frequently:
   ```bash
   pytest tests/ -v
   ```

4. Commit after each task completion

---

## Testing & Validation

### Run All Tests

```bash
# In each worktree
pytest tests/ -v --cov
```

### Run Benchmarks

```bash
# Performance comparison
python benchmarks/astar_benchmark.py
python benchmarks/semantic_benchmark.py
```

### Compare Against Baseline

To compare improvements vs. baseline:

```bash
# Baseline (main branch)
cd ../hound
./hound.py agent audit test_project --mode intuition

# With improvement (feature branch)
cd ../hound-astar-search
./hound.py agent audit test_project --mode intuition

# Compare metrics:
# - Time to first high-severity finding
# - Total hypotheses formed
# - Duplicate rate
# - Token usage
```

---

## A/B Testing Framework

Create test cases for comparing performance:

```python
# tests/benchmarks/ab_test.py
def benchmark_vulnerability_discovery(agent_config):
    """Measure time and accuracy of vulnerability discovery."""
    start = time.time()

    agent = AutonomousAgent(config=agent_config)
    results = agent.investigate(
        "Find all authentication vulnerabilities",
        max_iterations=20
    )

    elapsed = time.time() - start

    return {
        'time_seconds': elapsed,
        'hypotheses_found': len(results['hypotheses']),
        'high_severity': len([h for h in results['hypotheses'] if h['severity'] == 'high']),
        'duplicates': count_duplicates(results['hypotheses']),
        'tokens_used': results['token_usage']
    }


# Run A/B test
baseline = benchmark_vulnerability_discovery(baseline_config)
astar = benchmark_vulnerability_discovery(astar_config)

print(f"Speedup: {baseline['time_seconds'] / astar['time_seconds']:.1f}x")
print(f"Duplicates: {baseline['duplicates']} -> {astar['duplicates']} ({improvement}%)")
```

---

## Integration Strategy

After validating each improvement:

1. **Merge to main**:
   ```bash
   cd ../hound
   git merge feature/astar-search
   ```

2. **Update config.yaml.example** with new options

3. **Add feature flags** for gradual rollout:
   ```yaml
   agent:
     use_astar_search: false  # Default off until validated
   ```

4. **Document in CLAUDE.md**

5. **Add to CI/CD** test suite

---

## Priority & Dependencies

### Phase 1 (Independent - can be done in parallel)
- ✅ A* Search
- ✅ Semantic Deduplication
- ✅ Bayesian Confidence

### Phase 2 (Depends on Phase 1)
- Incremental Graphs (benefits from A* search)
- Attention Context (benefits from A* + semantic dedup)

### Phase 3 (Future)
- Game-theoretic planning
- Probabilistic programming
- Formal verification integration

---

## Success Metrics

Track these KPIs before/after each improvement:

| Metric | Baseline | Target |
|--------|----------|--------|
| Time to first high-severity vuln | 45s | 12s (73% faster) |
| Redundant node visits | 35% | 5% (86% reduction) |
| Duplicate hypotheses | 40% | 10% (75% reduction) |
| Context token usage | 150K | 60K (60% reduction) |
| False positive rate | 30% | 15% (50% reduction) |
| Coverage (LOC analyzed) | 65% | 90% (38% increase) |

---

## Support & Troubleshooting

### Common Issues

**Q: Sentence transformers download fails**
A: Pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

**Q: Tests fail with import errors**
A: Ensure you're in the correct worktree: `git worktree list`

**Q: Merge conflicts when integrating**
A: Rebase before merge: `git rebase main`

### Getting Help

1. Check test output: `pytest -v --tb=short`
2. Enable debug mode: `--debug` flag
3. Review plan for missed steps
4. Ask Claude for guidance (reference the plan file)

---

## Next Steps

1. **Choose an improvement** to start with (recommend A* search - high impact, moderate effort)
2. **Navigate to worktree**: `cd ../hound-astar-search`
3. **Execute the plan**: Use `superpowers:executing-plans` skill
4. **Run tests**: Validate each task
5. **Benchmark**: Measure performance improvements
6. **Repeat** for other improvements

Happy building! 🚀

---

## Academic References

For implementation details, see:

- **A* Search**: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (Ch. 3)
- **Semantic Similarity**: Reimers & Gurevych, "Sentence-BERT" (2019)
- **Bayesian Methods**: Gelman et al., "Bayesian Data Analysis" (Ch. 2-3)
- **Incremental Algorithms**: Ramalingam & Reps, "On the Computational Complexity of Dynamic Graph Problems" (1996)
- **Attention Mechanisms**: Vaswani et al., "Attention Is All You Need" (2017)
