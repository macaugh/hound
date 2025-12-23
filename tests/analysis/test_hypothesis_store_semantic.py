import pytest
from pathlib import Path
from analysis.concurrent_knowledge import HypothesisStore, Hypothesis


def test_hypothesis_store_rejects_semantic_duplicates(tmp_path):
    """Test that HypothesisStore rejects semantically similar hypotheses."""
    store_path = tmp_path / "hypotheses.json"
    store = HypothesisStore(store_path, agent_id="test")

    # Propose first hypothesis
    hyp1 = Hypothesis(
        title="Missing authentication in transfer function",
        description="The transfer function lacks authentication checks",
        vulnerability_type="access_control",
        severity="high",
        node_refs=["func_transfer"]
    )

    success1, hyp_id1 = store.propose(hyp1)
    assert success1 is True

    # Try to propose semantic duplicate (different wording, same issue)
    hyp2 = Hypothesis(
        title="Transfer function missing auth validation",
        description="No authentication validation in transfer",
        vulnerability_type="access_control",
        severity="high",
        node_refs=["func_transfer"]
    )

    success2, msg = store.propose(hyp2)
    assert success2 is False
    assert "semantic duplicate" in msg.lower() or "similar" in msg.lower()


def test_hypothesis_store_allows_different_nodes(tmp_path):
    """Test that similar text on different nodes is allowed."""
    store_path = tmp_path / "hypotheses.json"
    store = HypothesisStore(store_path, agent_id="test")

    # First hypothesis on func_transfer
    hyp1 = Hypothesis(
        title="Missing authentication check",
        description="Auth check missing",
        vulnerability_type="access_control",
        severity="high",
        node_refs=["func_transfer"]
    )
    store.propose(hyp1)

    # Similar text but different nodes
    hyp2 = Hypothesis(
        title="Missing authentication validation",
        description="Auth validation missing",
        vulnerability_type="access_control",
        severity="high",
        node_refs=["func_withdraw"]  # Different node
    )

    success, _ = store.propose(hyp2)
    # Should be allowed - different code location
    assert success is True
