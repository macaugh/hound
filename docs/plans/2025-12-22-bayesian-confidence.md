# Bayesian Confidence Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace arbitrary confidence scores (0.9="high", 0.6="medium") with proper Bayesian probability distributions that can be updated with evidence using Bayes' Rule.

**Architecture:** Implements Beta-Binomial conjugate priors for confidence modeling. Each hypothesis maintains alpha/beta parameters representing prior belief, updates using likelihood ratios from evidence, and provides credible intervals. Integrates into Hypothesis dataclass and HypothesisStore.

**Tech Stack:** scipy.stats (Beta distribution), numpy, existing Hypothesis infrastructure

---

## Quick Implementation Guide

### Dependencies

```txt
scipy>=1.11.0
numpy>=1.24.0
```

### Core Classes

```python
# analysis/hypothesis/bayesian_confidence.py
from dataclasses import dataclass
from scipy.stats import beta
import numpy as np


@dataclass
class BayesianConfidence:
    """Bayesian confidence with Beta distribution."""
    alpha: float = 2.0  # Prior successes + 1
    beta_param: float = 2.0  # Prior failures + 1

    @property
    def point_estimate(self) -> float:
        """Expected value (mean) of Beta distribution."""
        return self.alpha / (self.alpha + self.beta_param)

    def credible_interval(self, confidence=0.95) -> tuple[float, float]:
        """Bayesian credible interval."""
        lower = beta.ppf((1 - confidence) / 2, self.alpha, self.beta_param)
        upper = beta.ppf(1 - (1 - confidence) / 2, self.alpha, self.beta_param)
        return (float(lower), float(upper))

    def update_with_evidence(self, supports: bool, strength: float = 1.0):
        """Update distribution with new evidence.

        Args:
            supports: True if evidence supports hypothesis
            strength: Evidence strength (0-1)
        """
        if supports:
            self.alpha += strength
        else:
            self.beta_param += strength

    def from_legacy_score(score: float) -> 'BayesianConfidence':
        """Convert legacy 0-1 score to Bayesian representation."""
        # Map score to Beta parameters
        # score=0.9 -> Beta(9,1), score=0.5 -> Beta(1,1)
        alpha = max(1.0, score * 10)
        beta_p = max(1.0, (1 - score) * 10)
        return BayesianConfidence(alpha=alpha, beta_param=beta_p)
```

### Integration into Hypothesis

```python
# Modify analysis/concurrent_knowledge.py
@dataclass
class Hypothesis:
    # ... existing fields ...

    # Legacy field (keep for backward compat)
    confidence: float = 0.5

    # New Bayesian confidence (optional)
    bayesian_confidence: BayesianConfidence | None = None

    def __post_init__(self):
        # Initialize Bayesian from legacy if not provided
        if self.bayesian_confidence is None:
            self.bayesian_confidence = BayesianConfidence.from_legacy_score(
                self.confidence
            )
```

### Evidence Updates

```python
# New method in HypothesisStore
def add_evidence_bayesian(self,
                         hypothesis_id: str,
                         supports: bool,
                         strength: float = 1.0,
                         description: str = "") -> bool:
    """Add evidence and update Bayesian confidence."""
    def update(data):
        if hypothesis_id not in data["hypotheses"]:
            return data, False

        hyp = data["hypotheses"][hypothesis_id]

        # Update Bayesian confidence
        if 'bayesian_confidence' in hyp:
            bc = BayesianConfidence(**hyp['bayesian_confidence'])
            bc.update_with_evidence(supports, strength)
            hyp['bayesian_confidence'] = {
                'alpha': bc.alpha,
                'beta_param': bc.beta_param
            }

            # Update legacy confidence field
            hyp['confidence'] = bc.point_estimate

        # Add evidence record
        evidence_entry = {
            'description': description,
            'type': 'supports' if supports else 'refutes',
            'strength': strength,
            'created_at': datetime.now().isoformat()
        }
        hyp.setdefault('evidence', []).append(evidence_entry)

        return data, True

    return self.update_atomic(update)
```

### Tests

```python
# tests/analysis/test_bayesian_confidence.py
def test_bayesian_update_with_supporting_evidence():
    bc = BayesianConfidence(alpha=2, beta_param=2)
    initial = bc.point_estimate  # 0.5

    bc.update_with_evidence(supports=True, strength=1.0)

    assert bc.point_estimate > initial  # Should increase
    assert bc.alpha == 3.0

def test_bayesian_credible_interval():
    bc = BayesianConfidence(alpha=10, beta_param=2)

    lower, upper = bc.credible_interval(confidence=0.95)

    # High confidence (10 vs 2) should have narrow interval
    assert upper - lower < 0.3
    assert lower > 0.5  # Most mass above 0.5

def test_hypothesis_store_bayesian_evidence(tmp_path):
    store = HypothesisStore(tmp_path / "hyp.json", "agent")

    hyp = Hypothesis(
        title="Test vuln",
        description="Test",
        vulnerability_type="test",
        severity="medium"
    )

    _, hyp_id = store.propose(hyp)

    # Add supporting evidence
    store.add_evidence_bayesian(hyp_id, supports=True, strength=2.0)

    # Confidence should increase
    hyps = store.list_all()
    updated = next(h for h in hyps if h['id'] == hyp_id)
    assert updated['confidence'] > 0.5
```

### Configuration

```yaml
# config.yaml.example
hypothesis:
  # Bayesian confidence settings
  bayesian:
    enabled: true
    prior_alpha: 2.0  # Weak prior (2 successes)
    prior_beta: 2.0   # Weak prior (2 failures)
    evidence_strength_default: 1.0
```

## Testing & Validation

```bash
# Run tests
pytest tests/analysis/test_bayesian_confidence.py -v

# Benchmark
python benchmarks/bayesian_benchmark.py
```

## Migration Strategy

1. **Phase 1**: Add Bayesian fields alongside legacy confidence (backward compatible)
2. **Phase 2**: Update all new hypotheses to use Bayesian
3. **Phase 3**: Migrate existing hypotheses (batch script)
4. **Phase 4**: Deprecate legacy confidence field

## Expected Benefits

- **Proper uncertainty quantification**: 90% credible intervals instead of point estimates
- **Evidence accumulation**: Confidence updates correctly with Bayes' Rule
- **Scientific rigor**: Defensible probability claims
- **Better decision-making**: Know when to investigate further vs. reject

## Commit Messages

```
feat(hypothesis): add Bayesian confidence core classes
feat(hypothesis): integrate Bayesian updates into HypothesisStore
test(hypothesis): add Bayesian confidence test suite
docs(hypothesis): add Bayesian confidence documentation
```
