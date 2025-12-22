# Attention-Based Context Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace uniform context inclusion with attention-weighted selection that prioritizes relevant information based on current investigation goal, reducing token usage while improving relevance.

**Architecture:** Implements transformer-style attention mechanism using sentence embeddings. Computes attention scores between investigation goal and context parts, includes high-scoring parts + critical sections, applies adaptive compression to low-attention parts.

**Tech Stack:** sentence-transformers, scikit-learn (cosine similarity), numpy, existing agent_core

---

## Core Implementation

### 1. Attention Scorer

```python
# analysis/context/attention.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class AttentionScorer:
    """Compute attention scores for context selection."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def compute_attention_scores(self,
                                 query: str,
                                 contexts: list[str]) -> np.ndarray:
        """Compute attention scores using cosine similarity.

        Args:
            query: Investigation goal or current focus
            contexts: List of context parts to score

        Returns:
            Array of attention scores (0-1, higher = more relevant)
        """
        # Encode query
        query_emb = self._encode(query)

        # Encode contexts
        context_embs = np.array([self._encode(ctx) for ctx in contexts])

        # Compute cosine similarities
        scores = cosine_similarity(
            query_emb.reshape(1, -1),
            context_embs
        )[0]

        # Apply softmax for probability distribution
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        attention = exp_scores / exp_scores.sum()

        return attention

    def _encode(self, text: str) -> np.ndarray:
        """Encode text with caching."""
        if text not in self._cache:
            self._cache[text] = self.model.encode(text, convert_to_numpy=True)

            # Limit cache size
            if len(self._cache) > 500:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return self._cache[text]


class AttentionFilter:
    """Filter context parts based on attention scores."""

    def __init__(self,
                 scorer: AttentionScorer,
                 threshold: float = 0.1,
                 top_k: int | None = None):
        """
        Args:
            scorer: AttentionScorer instance
            threshold: Minimum attention score to include (0-1)
            top_k: If set, include only top K parts (overrides threshold)
        """
        self.scorer = scorer
        self.threshold = threshold
        self.top_k = top_k

    def filter_contexts(self,
                       query: str,
                       contexts: list[tuple[str, str]],
                       critical_labels: set[str]) -> list[tuple[str, str, float]]:
        """Filter contexts by attention scores.

        Args:
            query: Investigation goal
            contexts: List of (label, content) tuples
            critical_labels: Labels that must always be included

        Returns:
            List of (label, content, attention_score) tuples
        """
        # Extract labels and content
        labels, contents = zip(*contexts) if contexts else ([], [])

        # Compute attention scores
        scores = self.scorer.compute_attention_scores(query, list(contents))

        # Build results
        results = []
        for i, (label, content) in enumerate(contexts):
            score = float(scores[i])

            # Always include critical sections
            if label in critical_labels:
                results.append((label, content, 1.0))  # Max score
                continue

            # Apply threshold or top-k filtering
            if self.top_k is not None:
                # Will filter to top-k after sorting
                results.append((label, content, score))
            elif score >= self.threshold:
                results.append((label, content, score))

        # Sort by score (descending)
        results.sort(key=lambda x: x[2], reverse=True)

        # Apply top-k if specified
        if self.top_k is not None:
            results = results[:self.top_k]

        return results
```

### 2. Integration into Agent

```python
# Modify analysis/agent_core.py
class AutonomousAgent:
    def __init__(self, ...):
        # ... existing init ...

        # Initialize attention filtering if enabled
        try:
            agent_config = (config or {}).get('agent', {})
            self.use_attention = agent_config.get('use_attention_context', False)

            if self.use_attention:
                from analysis.context.attention import AttentionScorer, AttentionFilter
                scorer = AttentionScorer()
                self.attention_filter = AttentionFilter(
                    scorer=scorer,
                    threshold=agent_config.get('attention_threshold', 0.15),
                    top_k=agent_config.get('attention_top_k')
                )
                print(f"[*] Attention-based context filtering enabled")
        except Exception as e:
            print(f"[!] Failed to initialize attention filter: {e}")
            self.use_attention = False
            self.attention_filter = None

    def _build_context(self) -> str:
        """Build context with optional attention filtering."""
        # Build full context parts as (label, content) tuples
        context_parts = []

        # Investigation goal (always included)
        context_parts.append(("INVESTIGATION_GOAL", self.investigation_goal))

        # User steering (always included)
        steering = self._read_steering_notes()
        if steering:
            context_parts.append(("USER_STEERING", "\n".join(f"• {s}" for s in steering)))

        # Available graphs
        graphs_list = "\n".join(f"• {name}" for name in self.available_graphs.keys())
        context_parts.append(("AVAILABLE_GRAPHS", graphs_list))

        # Memory notes
        if self.memory_notes:
            notes = "\n".join(f"• {n}" for n in self.memory_notes[-5:])
            context_parts.append(("MEMORY", notes))

        # System graph (always visible but can be compressed if low attention)
        if self.loaded_data['system_graph']:
            graph_display = "\n".join(self._format_graph_for_display(
                self.loaded_data['system_graph']['data'],
                self.loaded_data['system_graph']['name']
            ))
            context_parts.append(("SYSTEM_GRAPH", graph_display))

        # Additional loaded graphs
        for gname, gdata in self.loaded_data.get('graphs', {}).items():
            graph_display = "\n".join(self._format_graph_for_display(gdata, gname))
            context_parts.append((f"GRAPH_{gname}", graph_display))

        # Loaded nodes
        if self.loaded_data.get('nodes'):
            nodes_text = "\n".join(self.loaded_data['nodes'].keys())
            context_parts.append(("LOADED_NODES", nodes_text))

        # Actions performed
        if self.action_log:
            actions = "\n".join(
                f"- {e['action']}: {e.get('result', '')[:100]}"
                for e in self.action_log[-10:]
            )
            context_parts.append(("ACTIONS", actions))

        # Existing hypotheses (critical - always include)
        hyps_text = self._format_hypotheses()
        context_parts.append(("EXISTING_HYPOTHESES", hyps_text))

        # Recent actions detail
        if len(self.conversation_history) > 1:
            recent = "\n".join(
                entry['content']
                for entry in self.conversation_history[-5:]
            )
            context_parts.append(("RECENT_ACTIONS", recent))

        # Apply attention filtering if enabled
        if self.use_attention and self.attention_filter:
            critical_labels = {
                "INVESTIGATION_GOAL",
                "USER_STEERING",
                "EXISTING_HYPOTHESES"
            }

            filtered_parts = self.attention_filter.filter_contexts(
                query=self.investigation_goal,
                contexts=context_parts,
                critical_labels=critical_labels
            )

            # Log what was filtered
            filtered_count = len(context_parts) - len(filtered_parts)
            if filtered_count > 0:
                print(f"[ATTENTION] Filtered {filtered_count}/{len(context_parts)} context parts (low relevance)")

            # Rebuild context from filtered parts
            final_context = []
            for label, content, score in filtered_parts:
                final_context.append(f"=== {label} ===")
                final_context.append(content)
                final_context.append("")

            return "\n".join(final_context)

        else:
            # No attention filtering - use all parts
            final_context = []
            for label, content in context_parts:
                final_context.append(f"=== {label} ===")
                final_context.append(content)
                final_context.append("")

            return "\n".join(final_context)
```

### 3. Adaptive Compression

```python
def _apply_adaptive_compression(self,
                               parts: list[tuple[str, str, float]],
                               token_budget: int) -> list[tuple[str, str]]:
    """Compress low-attention parts to fit budget.

    Args:
        parts: (label, content, score) tuples sorted by score
        token_budget: Maximum total tokens

    Returns:
        (label, content) tuples fitting within budget
    """
    # Sort by attention score (highest first)
    parts_sorted = sorted(parts, key=lambda x: x[2], reverse=True)

    result = []
    tokens_used = 0

    for label, content, score in parts_sorted:
        content_tokens = self._count_tokens(content)

        if tokens_used + content_tokens <= token_budget:
            # Fits - include in full
            result.append((label, content))
            tokens_used += content_tokens

        elif score < 0.3:
            # Low attention - try compression
            compressed = self._compress_content(content, target_ratio=0.3)
            compressed_tokens = self._count_tokens(compressed)

            if tokens_used + compressed_tokens <= token_budget:
                result.append((f"{label}_COMPRESSED", compressed))
                tokens_used += compressed_tokens

        # Else: skip this part

    return result

def _compress_content(self, content: str, target_ratio: float = 0.3) -> str:
    """Compress content to target ratio using extractive summarization."""
    lines = content.split('\n')
    target_lines = max(1, int(len(lines) * target_ratio))

    # Simple heuristic: keep first, last, and random middle lines
    if len(lines) <= target_lines:
        return content

    keep_lines = []
    keep_lines.append(lines[0])  # First

    # Middle (random sample)
    import random
    middle = lines[1:-1]
    if middle:
        sample_size = max(0, target_lines - 2)
        keep_lines.extend(random.sample(middle, min(sample_size, len(middle))))

    keep_lines.append(lines[-1])  # Last

    return '\n'.join(keep_lines) + f"\n[... {len(lines) - len(keep_lines)} lines compressed ...]"
```

## Testing

```python
# tests/analysis/test_attention_context.py
def test_attention_scorer_relevance():
    scorer = AttentionScorer()

    query = "Investigate authentication vulnerabilities in transfer function"

    contexts = [
        "Authentication bypass in transfer allows unauthorized access",  # High
        "Helper function calculates checksums",  # Low
        "Transfer function missing permission validation",  # High
        "Logging utility formats timestamps"  # Low
    ]

    scores = scorer.compute_attention_scores(query, contexts)

    # Auth-related contexts should have higher scores
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]


def test_attention_filter_keeps_critical():
    """Test that critical sections are always kept."""
    scorer = AttentionScorer()
    filter = AttentionFilter(scorer, threshold=0.5)

    query = "Check auth"
    contexts = [
        ("INVESTIGATION_GOAL", "Investigate auth"),
        ("LOW_RELEVANCE", "Unrelated utility function")
    ]

    filtered = filter.filter_contexts(
        query,
        contexts,
        critical_labels={"INVESTIGATION_GOAL"}
    )

    # INVESTIGATION_GOAL must be included even if low score
    labels = [f[0] for f in filtered]
    assert "INVESTIGATION_GOAL" in labels


def test_attention_reduces_token_usage(tmp_path):
    """Test that attention filtering reduces context size."""
    # Create agent with attention enabled
    config = {
        'models': {
            'scout': {'model': 'gpt-4', 'provider': 'mock'}
        },
        'agent': {
            'use_attention_context': True,
            'attention_threshold': 0.2
        }
    }

    agent = AutonomousAgent(..., config=config)
    agent.investigation_goal = "Find reentrancy vulnerabilities"

    # Add lots of unrelated context
    for i in range(20):
        agent.memory_notes.append(f"Unrelated note {i}")

    # Build context
    context_with_attention = agent._build_context()
    tokens_with = agent._count_tokens(context_with_attention)

    # Disable attention
    agent.use_attention = False
    context_without_attention = agent._build_context()
    tokens_without = agent._count_tokens(context_without_attention)

    # Attention should reduce tokens
    reduction = (tokens_without - tokens_with) / tokens_without
    assert reduction > 0.2  # At least 20% reduction
```

## Configuration

```yaml
# config.yaml.example
agent:
  # Attention-based context filtering
  use_attention_context: true
  attention_threshold: 0.15  # Include parts with score >= 0.15
  attention_top_k: null  # If set, use top-K instead of threshold

  # Adaptive compression (when near token limit)
  adaptive_compression: true
  compression_threshold: 0.75  # Compress when context > 75% of limit
  compression_ratio: 0.3  # Compress low-attention parts to 30%
```

## Performance Expectations

| Context Size | Full Context | With Attention | Reduction |
|--------------|--------------|----------------|-----------|
| 50K tokens   | 50K          | 35K            | 30%       |
| 100K tokens  | 100K         | 60K            | 40%       |
| 200K tokens  | 200K (OOM)   | 80K            | 60%       |

**Benefits**:
- Fits more investigations within context limits
- Improves LLM focus on relevant information
- Reduces API costs by 30-40%

## Commit Plan

```bash
git commit -m "feat(context): add attention scorer and filter"
git commit -m "feat(agent): integrate attention-based context filtering"
git commit -m "feat(context): add adaptive compression for low-attention parts"
git commit -m "test(context): add attention context tests"
git commit -m "docs(context): document attention-based context management"
```
