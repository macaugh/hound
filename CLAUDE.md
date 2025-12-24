# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hound is an autonomous AI-powered security auditing system that uses dynamic knowledge graphs and LLM agents to find vulnerabilities in code. It employs a "junior/senior" agent pattern where lightweight scout models explore code and heavyweight strategist models perform deep reasoning.

**Key concept**: Hound doesn't look for bugs directly during exploration—it builds understanding through graphs, observations, and hypotheses that evolve with confidence scores over time.

## Essential Commands

### Development Workflow
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest                           # All tests
pytest tests/test_agent_core.py  # Single test file
pytest -v --tb=short             # Verbose with short tracebacks
pytest -m "not slow"             # Skip slow tests
pytest --cov                     # With coverage

# Linting
ruff check .                     # Check for issues
ruff check --fix .               # Auto-fix issues
```

### Running Hound
```bash
# Main entry point
./hound.py <command>

# Common workflows
./hound.py project create myaudit /path/to/code
./hound.py graph build myaudit --auto --files "src/A.sol,src/B.sol"
./hound.py agent audit myaudit --mode sweep
./hound.py agent audit myaudit --mode intuition --time-limit 300
./hound.py finalize myaudit
./hound.py report myaudit

# With telemetry UI
./hound.py agent audit myaudit --telemetry --debug
python chatbot/run.py  # In separate terminal
```

## Architecture

### Core Components

**Entry Point**
- `hound.py`: CLI application using Typer with subcommand groups (project, agent, graph, poc)
- Routes commands to `commands/` modules

**Command Layer** (`commands/`)
- `project.py`: Project management (create, list, info, coverage, hypotheses)
- `agent.py`: Agent execution (audit modes: sweep, intuition)
- `graph.py`: Knowledge graph building and management
- `finalize.py`: QA pass over hypotheses with reasoning models
- `report.py`: HTML report generation
- `poc.py`: Proof-of-concept management

**Analysis Core** (`analysis/`)
- `agent_core.py`: Autonomous agent that makes decisions via structured tool calls (load_graph, load_nodes, update_node, form_hypothesis, complete)
- `strategist.py`: Strategic planning and hypothesis formation using heavyweight models
- `graph_builder.py`: Dynamic graph construction with agent-driven schema discovery
- `concurrent_knowledge.py`: Thread-safe stores (GraphStore, HypothesisStore, CoverageIndex)
- `session_tracker.py`: Per-audit session management with planning state
- `plan_store.py`: Per-session investigation planning with statuses (planned/in_progress/done/dropped)
- `report_generator.py`: Professional HTML report generation

**LLM Integration** (`llm/`)
- `unified_client.py`: Multi-provider LLM client with profile-based configuration
- Provider implementations: OpenAI, Anthropic, Gemini (AI Studio + Vertex AI), DeepSeek, xAI
- `token_tracker.py`: Per-model token usage tracking
- `tokenization.py`: Model-specific token counting

**Data Ingestion** (`ingest/`)
- `manifest.py`: Repository scanning and file card creation
- `bundles.py`: Adaptive code bundling for context management

**Project Storage**
- Projects stored in `~/.hound/projects/<project_name>/`
- Structure: `graphs/`, `sessions/`, `hypothesis_store.json`, `coverage.json`, `source/`

### Key Patterns

**1. Dynamic Knowledge Graphs**
- Agents propose graph schemas based on codebase (not hardcoded)
- Common graphs: SystemArchitecture, StateMutationGraph, InterContractCallGraph, AuthorizationRolesMap
- Nodes have observations (verified facts) and assumptions (unverified)
- Graphs refine iteratively, adding confidence scores and evidence

**2. Hypothesis Evolution**
- Not "bug reports" but evolving theories with confidence scores (0.0-1.0)
- Statuses: proposed → investigating → confirmed/rejected
- Stored persistently across sessions in HypothesisStore
- Agent updates hypotheses as it gathers evidence

**3. Concurrent File Stores**
- Base class: `ConcurrentFileStore` with file locking (portalocker)
- Subclasses: HypothesisStore, GraphStore, CoverageIndex, PlanStore
- Pattern: `with store.lock(): data = store.read(); ...; store.write(data)`
- All stores are thread-safe for parallel agent execution

**4. Agent Decision Loop**
- Agent receives context (graphs, coverage, hypotheses)
- Returns structured `AgentDecision` with action/reasoning/parameters
- Actions are tool calls: load_graph, load_nodes, update_node, form_hypothesis, update_hypothesis, complete
- No prescriptive flow—agent decides next steps autonomously

**5. Session Planning**
- Each audit run has a session with a PlanStore
- Strategist creates investigation plans (titles + descriptions)
- Plans have statuses: planned → in_progress → done/dropped/superseded
- On resume, stale in_progress items reset to planned

**6. Model Profiles**
- Config-driven model selection via profiles: scout, strategist, graph, finalize, lightweight, reporting
- Each profile can specify provider, model, max_context, reasoning_effort
- Examples: scout=gpt-5-mini (exploration), strategist=gpt-5 (deep reasoning), graph=gemini-2.5-pro (large context)

### Configuration

**config.yaml** (copy from `config.yaml.example`):
- API keys via environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- Model profiles with provider, model name, max_context, reasoning_effort
- Context limits: models can override global `context.max_tokens`
- Gemini supports both AI Studio (API key) and Vertex AI (ADC/service account)

**Environment Variables**:
```bash
OPENAI_API_KEY          # Required for OpenAI models
OPENAI_BASE_URL         # Optional: custom endpoint
ANTHROPIC_API_KEY       # For Claude models
GOOGLE_API_KEY          # For Gemini AI Studio
GOOGLE_USE_VERTEX_AI=1  # Enable Vertex AI for Gemini
VERTEX_PROJECT_ID       # GCP project for Vertex AI
VERTEX_LOCATION         # Region (us-central1, etc.)
```

## Development Guidelines

### Module Import Quirks
- Project uses local `llm/` package that conflicts with global `llm` package
- `hound.py` and `tests/conftest.py` have sys.path hacks to prioritize local modules
- When importing: `from llm.unified_client import UnifiedLLMClient` (not `hound.llm`)

### Testing
- Uses pytest with custom conftest.py to resolve local llm modules
- Mock provider available: `llm.mock_provider.MockProvider` for testing without API calls
- Integration tests marked with `@pytest.mark.integration`
- Slow tests marked with `@pytest.mark.slow` (skip with `-m "not slow"`)

### Code Style
- Line length: 100 (pyproject.toml), 120 (ruff.toml) - use 100 for new code
- Python 3.10+ (uses union syntax: `str | None`)
- Ruff for linting (E, F, I, UP rules)
- Type hints encouraged but not strictly enforced

### Adding New LLM Providers
1. Create provider class in `llm/<provider>_provider.py` inheriting from `BaseProvider`
2. Implement `call()` method returning `LLMResponse`
3. Register in `unified_client.py`'s provider mapping
4. Add config section in `config.yaml.example`

### Adding New Agent Actions
1. Add parameters to `AgentParameters` schema in `agent_core.py`
2. Implement handler method in `AutonomousAgent` class
3. Update system prompt examples in `agent_core.py`
4. Add to decision JSON schema description

### Debug Mode
- Enable with `--debug` flag on any agent command
- Saves all LLM interactions to `.hound_debug/` as HTML reports
- Includes full prompts, responses, and timestamps
- Uses `analysis/debug_logger.py`

### Handling Long Audit Outputs
- Audit output data can exceed Claude Sonnet's context limits
- When examining task outputs or audit results, prefer using `head` to view truncated output
- Example: `head -n 100 file.txt` or use `Read` tool with `limit` parameter
- This prevents context overflow while still allowing inspection of results

## Validation Framework

### Overview

The validation framework adds skeptical analysis to hypothesis finalization, automatically rejecting common false positive patterns. This multi-stage validation process runs during `hound.py finalize` to ensure only legitimate security vulnerabilities are confirmed.

The framework consists of three analysis stages:

1. **Permission Analysis**: Detects admin-only vulnerabilities (foot-guns) by tracing function modifiers and access control
2. **Impact Classification**: Distinguishes security issues from compatibility issues and code quality concerns
3. **Pattern Matching**: Matches against known false positive patterns using keyword indicators

### False Positive Patterns

The framework recognizes four common categories of false positives:

**Admin Foot-Gun**: Only admin can trigger the vulnerability
- Detects: `onlyOwner`, `onlyAdmin`, `diamondCut`, `facet`, `upgrade` modifiers
- Example: "Removing liquidator facet causes DoS" - Only admin can remove facets via diamondCut
- Verdict: REJECT (not a vulnerability, requires trusted admin to act maliciously)

**Sybil Resistance**: "User can create multiple addresses"
- Detects: "one per address", "multiple addresses", "create new wallet", "bypass one loan"
- Example: "Bypass one loan per owner restriction" - Each address still has one loan limit
- Verdict: REJECT (fundamental blockchain limitation, not a vulnerability)

**Mathematical Noise**: Wei-level precision loss
- Detects: "dust", "wei", "precision loss", "division truncation", "rounding", "1-2 wei"
- Example: "Integer division loses 1-2 wei on large transfers" - Economically zero impact
- Verdict: REJECT (inherent to Solidity, affects all smart contracts)

**Design Limitation**: Unbounded arrays with pagination alternatives
- Detects: "unbounded array", "getAllLoans", "view function", "gas limit", "pagination"
- Example: "getAllLoans can revert due to gas limit" - Paginated getLoans() function exists
- Verdict: WARN or REJECT (not exploitable if pagination available)

### Configuration

Enable or disable validation stages in `config.yaml`:

```yaml
validation:
  enable_permission_analysis: true    # Trace function permissions
  enable_impact_classification: true  # Classify impact type
  enable_pattern_matching: true       # Match false positive patterns

  disqualify_on:
    admin_only: true                  # Reject admin-only issues
    compatibility_issue: true         # Reject compatibility concerns
    quality_issue: true               # Reject code quality issues
    matched_pattern: true             # Reject matched patterns
```

Configuration is loaded from `config.yaml` or defaults to all stages enabled.

### Usage

Validation runs automatically during the finalization stage:

```bash
# Run finalization with validation
./hound.py finalize myproject --threshold 0.6

# Output shows validation stages for each hypothesis
Reviewing: hasNoLoan Bypass (MEDIUM)
  ✓ Code Review: Pattern present in contract
  ✗ Permission Analysis: Anyone can create addresses
  ✗ Pattern Match: sybil_resistance
  → VERDICT: REJECTED (Sybil resistance limitation)

# Summary includes rejection breakdown
Rejection Breakdown:
  • Pattern Match (sybil_resistance): 3
  • Permission Analysis (admin_only): 2
  • Impact Classification (quality_issue): 1
```

The output clearly shows:
- Which validation stage rejected the hypothesis
- Which specific pattern matched
- The reason for rejection

### Implementation Files

The validation framework is implemented across several modules:

**Core Validation Components** (`analysis/`):
- `permission_tracer.py`: Traces function modifiers and access control to detect admin-only vulnerabilities
- `impact_classifier.py`: Classifies hypotheses as security, compatibility, or quality issues
- `false_positive_patterns.py`: Matches against known false positive patterns using keyword indicators

**Integration** (`commands/`):
- `finalize.py`: Integrates validation stages into hypothesis review process
- Shows validation stage details in rejection output
- Provides rejection breakdown summary

### Extending Patterns

To add new false positive patterns, edit `analysis/false_positive_patterns.py`:

```python
# Add to PATTERNS list in FalsePositivePatternMatcher class
PATTERNS = [
    {
        'name': 'your_pattern_name',
        'indicators': ['keyword1', 'keyword2', 'phrase to detect'],
        'description': 'Brief description of what this pattern represents',
        'rule': 'Explanation of why this disqualifies the hypothesis',
        'severity': 'DISQUALIFY'  # or 'WARN' for non-critical patterns
    }
]
```

Pattern matching uses case-insensitive keyword matching across hypothesis title, description, and vulnerability type. Indicators should be specific enough to avoid false matches while covering common variations of the pattern.

### Testing Validation

Test the validation framework with:

```bash
# Run validation tests
pytest tests/test_permission_tracer.py
pytest tests/test_impact_classifier.py
pytest tests/test_false_positive_patterns.py

# Run end-to-end validation tests
pytest tests/test_validation_e2e.py
```

Tests include real-world examples from security audits to verify pattern matching accuracy.

## Common Patterns

### Reading a Graph
```python
from analysis.concurrent_knowledge import GraphStore
store = GraphStore(project_dir / "graphs")
with store.lock():
    graphs = store.read()
    graph = graphs.get("SystemArchitecture")
```

### Updating Hypotheses
```python
from analysis.concurrent_knowledge import HypothesisStore
hyp_store = HypothesisStore(project_dir)
with hyp_store.lock():
    hypotheses = hyp_store.read()
    hypotheses.append({
        "id": "hyp_12345",
        "description": "...",
        "confidence": 0.7,
        "status": "proposed"
    })
    hyp_store.write(hypotheses)
```

### Using UnifiedLLMClient
```python
from llm.unified_client import UnifiedLLMClient
from utils.config_loader import load_config

config = load_config()
client = UnifiedLLMClient(cfg=config, profile="scout")
response = client.call(
    messages=[{"role": "user", "content": "Analyze this code..."}],
    temperature=0.7
)
```

### Token Counting
```python
from llm.tokenization import count_tokens
tokens = count_tokens(text, model_name="gpt-4o")
```

## Testing Patterns

### Using Mock Provider
```python
from llm.mock_provider import MockProvider
provider = MockProvider(responses=["Test response"])
response = provider.call(messages=[...])
```

### Temporary Project Setup
```python
import tempfile
from pathlib import Path
from commands.project import ProjectManager

with tempfile.TemporaryDirectory() as tmpdir:
    pm = ProjectManager(base_dir=Path(tmpdir))
    project_dir = pm.create_project("test", source_path)
```

## Important Constraints

- **Context limits**: Graph building uses large context models (1M+ tokens), but agents use smaller contexts (256k)
- **File locking**: Always use `with store.lock()` when accessing concurrent stores
- **Session state**: Sessions track coverage and planning; reusing session IDs continues from previous state
- **Hypothesis visibility**: Hypotheses are global by default (persist across sessions)
- **Graph iteration**: Graphs refine iteratively; more iterations = better quality but higher cost
