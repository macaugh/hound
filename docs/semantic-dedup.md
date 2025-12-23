# Semantic Duplicate Detection

## Overview

Hound uses semantic similarity matching to prevent duplicate vulnerability hypotheses. This goes beyond exact title matching to catch duplicates with different wording.

## How It Works

### Sentence Transformers

Uses `all-MiniLM-L6-v2` model to encode hypothesis titles into 384-dimensional vectors. These embeddings capture semantic meaning, allowing the system to recognize that "Missing authentication" and "Auth check missing" refer to the same concept.

### Duplicate Detection Algorithm

```python
def is_duplicate(new_hyp, existing_hyps):
    for existing in existing_hyps:
        text_similarity = cosine_similarity(
            embed(new_hyp.title),
            embed(existing.title)
        )

        node_overlap = jaccard_similarity(
            new_hyp.nodes,
            existing.nodes
        )

        if text_similarity >= 0.85 AND node_overlap >= 0.3:
            return True
    return False
```

### Thresholds

- **Similarity threshold**: 0.85 (85% similar text)
- **Node overlap**: 0.3 (30% shared nodes via Jaccard similarity)

**Both conditions must be met** to reject as duplicate. This prevents false positives where similar text describes issues in different code locations.

## Examples

### Detected Duplicates

✅ **Original**: "Missing authentication in transfer function"
❌ **Duplicate**: "Transfer function missing auth validation"
- Similarity: 0.91, Node overlap: 100% (both reference `func_transfer`)

✅ **Original**: "Reentrancy in withdraw"
❌ **Duplicate**: "Withdraw function vulnerable to reentrancy attack"
- Similarity: 0.88, Node overlap: 100%

### Allowed (Not Duplicates)

✅ "Missing auth in transfer" (nodes: `func_transfer`)
✅ "Missing auth in withdraw" (nodes: `func_withdraw`)
- Similarity: 0.92, **Node overlap: 0%** → Different code locations, allowed

✅ "Integer overflow in balance"
✅ "Missing permission check in transfer"
- **Similarity: 0.42** → Different vulnerability types, allowed

## Performance

- **Model load time**: ~2 seconds (first use only, lazy-loaded)
- **Embedding cache**: 1000 entries with FIFO eviction (evicts to 800 when full)
- **Comparison time**: <5ms per hypothesis
- **Memory usage**: ~92MB (model + cache)
- **Thread-safe**: Cache uses locking for concurrent access

## Configuration

Edit `config.yaml`:

```yaml
hypothesis:
  semantic_matching:
    enabled: true
    model: 'all-MiniLM-L6-v2'        # Sentence transformer model
    similarity_threshold: 0.85        # 0-1 (higher = stricter)
    node_overlap_threshold: 0.3       # 0-1 (higher = stricter)
```

### Tuning Thresholds

**If you see false positives** (legitimate hypotheses rejected):
- Lower `similarity_threshold` to 0.80 or 0.75
- Lower `node_overlap_threshold` to 0.2

**If you see false negatives** (duplicates slipping through):
- Raise `similarity_threshold` to 0.90
- Raise `node_overlap_threshold` to 0.4 or 0.5

## Architecture

### Components

**`analysis/hypothesis/semantic_matcher.py`**
- `SimilarityMatcher`: Protocol defining the interface
- `SemanticMatcher`: Main implementation using sentence-transformers
- `DummyMatcher`: Fallback that never matches (graceful degradation)
- `compute_node_overlap()`: Jaccard similarity for node lists
- `is_duplicate_hypothesis()`: Combined text + node checking

**`analysis/concurrent_knowledge.py`**
- `HypothesisStore.propose()`: Integration point
- Lazy initialization of `SemanticMatcher`
- Graceful fallback if model loading fails

### Error Handling

If semantic matching fails (model download issues, missing dependencies, etc.):
1. System logs warning but continues functioning
2. Falls back to `DummyMatcher` that never rejects
3. Exact title matching still works
4. No cascading failures

## Troubleshooting

### Model Download Issues

If `sentence-transformers` fails to download model:

```bash
# Pre-download model manually
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Model Cache Location

Models are cached in:
- Linux: `~/.cache/huggingface/`
- macOS: `~/.cache/huggingface/`
- Windows: `C:\Users\<username>\.cache\huggingface\`

### Memory Issues

If running on constrained systems:
- Model requires ~90MB RAM
- Cache adds ~1.5MB per 1000 embeddings
- Can disable by setting `enabled: false` in config

### Testing Semantic Matching

Run the test suite:

```bash
# Test semantic matcher
pytest tests/analysis/hypothesis/test_semantic_matcher.py -v

# Test HypothesisStore integration
pytest tests/analysis/test_hypothesis_store_semantic.py -v

# Run all tests
pytest tests/analysis/ -v
```

## Implementation Details

### Caching Strategy

Embeddings are cached to avoid recomputing:
- Key: hypothesis title (string)
- Value: 384-dimensional embedding (numpy array)
- Max size: 1000 entries
- Eviction: FIFO to 800 entries when full
- Thread-safe: Uses `threading.Lock`

### Node Overlap (Jaccard Similarity)

```python
overlap = len(nodes1 ∩ nodes2) / len(nodes1 ∪ nodes2)
```

Example:
- `nodes1 = ['func_transfer', 'func_validate']`
- `nodes2 = ['func_transfer', 'func_authorize']`
- Overlap = 1 / 3 = 0.33 ✓ (above 0.3 threshold)

### Type Safety

Uses Protocol-based typing for flexibility:

```python
class SimilarityMatcher(Protocol):
    threshold: float
    def compute_similarity(self, text1: str, text2: str) -> float: ...
```

This allows:
- Type checking without inheritance
- Easy mocking in tests
- Alternative matcher implementations

## Best Practices

1. **Keep default thresholds** unless you have specific issues
2. **Monitor false positives** in logs (look for "Semantic duplicate of...")
3. **Pre-download model** on production systems before first use
4. **Test with real data** to tune thresholds for your codebase
5. **Review rejected hypotheses** periodically to ensure quality

## Future Enhancements

Potential improvements:
- **Approximate nearest neighbor search** (FAISS) for large hypothesis sets
- **Batch comparison** for better performance
- **Custom embedding models** trained on vulnerability descriptions
- **Configurable similarity algorithms** (cosine vs. dot product)
- **Learning-based threshold adjustment** based on user feedback
