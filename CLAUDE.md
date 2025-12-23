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

### Semantic Duplicate Detection

Prevents duplicate vulnerability hypotheses using semantic similarity:
- **Location**: `analysis/hypothesis/semantic_matcher.py`
- **Model**: Uses `all-MiniLM-L6-v2` sentence transformer for embeddings
- **Algorithm**: Combines text similarity (cosine) + node overlap (Jaccard)
- **Thresholds**: 85% text similarity + 30% node overlap (both required)
- **Integration**: Automatically runs in `HypothesisStore.propose()`
- **Thread-safe**: Cache uses locking for concurrent access
- **Performance**: ~2s model load (lazy), <5ms per comparison

**Configuration** (`config.yaml`):
```yaml
hypothesis:
  semantic_matching:
    enabled: true
    similarity_threshold: 0.85  # 0-1, higher = stricter
    node_overlap_threshold: 0.3  # 0-1, higher = stricter
```

**Test semantic matching**:
```bash
pytest tests/analysis/hypothesis/ -v
pytest tests/analysis/test_hypothesis_store_semantic.py -v
```

**Documentation**: See `docs/semantic-dedup.md` for details, examples, and tuning guide.

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
