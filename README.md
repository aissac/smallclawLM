# SmallClawLM

**Zero-token AI agent powered by Google NotebookLM.**

One agent, one notebook, one specialty. Each SmallClawLM agent owns exactly one NotebookLM notebook — the notebook's sources define its domain expertise. All reasoning runs through Gemini inside NotebookLM. **No external LLM API keys needed.**

## Architecture: Hybrid Fast/Slow Path

SmallClawLM uses an intent router to classify user input:

- **Fast Path (Pipeline):** Direct tool calls for known intents — podcast, report, quiz, research, etc. No LLM reasoning needed. Saves rate limits for actual work.
- **Slow Path (Agent):** Full CodeAgent with NLMModel brain for conversational, analytical, or multi-step tasks that need reasoning.

```
User Input
    │
    ▼
┌──────────┐     ┌──────────────────┐
│  Router   │────►│ Fast Path: Pipeline │──► Direct API Call ──► Result
│ (intent)  │     └──────────────────┘
│           │────►┌──────────────────┐
│           │     │ Slow Path: Agent   │──► NLMModel.chat ──► CodeAgent ──► Result
└──────────┘     └──────────────────┘
```

## Agent Specialties

| Specialty | Tools | Use Case |
|-----------|-------|----------|
| research | deep_research, ask_notebook, add_source, list_sources | Web research and Q&A |
| podcast | ask_notebook, add_source, generate_podcast | Audio overviews |
| quiz | ask_notebook, add_source, generate_quiz | Study aids |
| report | ask_notebook, add_source, generate_report | Structured reports |
| mindmap | ask_notebook, add_source, generate_mind_map | Visual summaries |
| all | All tools | General purpose |

## Install

```bash
pip install smallclawlm
# Or from source:
git clone https://github.com/aissac/smallclawLM.git
cd smallclawLM
pip install -e .
```

## Authenticate

```bash
smallclaw login
# Opens browser for Google authentication
```

## Usage

### CLI — Auto-routed (recommended)

```bash
# Auto-routes to fast path (direct podcast generation)
smallclaw run "generate a podcast"

# Auto-routes to fast path (deep research)
smallclaw run "research fusion energy breakthroughs"

# Auto-routes to slow path (conversational reasoning)
smallclaw run "Why is cold fusion so difficult to achieve?"

# Force slow path (agent reasoning)
smallclaw run "explain quantum computing" --force-slow

# Force fast path
smallclaw run "generate report" --force-fast

# Verbose routing info
smallclaw run "research fusion energy" -v
```

### CLI — Shortcut Commands (fast path only)

```bash
smallclaw research "mRNA vaccine technology"
smallclaw podcast --notebook-id abc123
smallclaw report --notebook-id abc123
smallclaw quiz --notebook-id abc123
smallclaw mindmap --notebook-id abc123
smallclaw list-sources --notebook-id abc123
```

### CLI — Pipeline

```bash
# Chain operations without an agent
smallclaw pipe --add-source "https://arxiv.org/abs/2401.12345" --generate podcast
smallclaw pipe --research "climate change" --generate report
smallclaw pipe --ask "What are the key findings?" --notebook-id abc123
```

### CLI — Interactive Agent

```bash
# Start a slow-path session
smallclaw agent --notebook-id abc123
```

### Python API

```python
from smallclawlm import NLMAgent, create_agent, route

# Auto-route: fast or slow path
result = route("generate a podcast")
# RouteResult(path=Path.FAST, intent="generate_podcast", confidence=0.9, params={})

# Fast path: direct Pipeline call
from smallclawlm.extensions.pipeline import Pipeline, ArtifactType
pipe = Pipeline(notebook_id="abc123")
pipe.research("fusion energy")
result = pipe.execute()

# Slow path: full agent
agent = create_agent("research", notebook_id="abc123")
result = agent.run("Why is cold fusion difficult to achieve?")
```

## Design Decisions

- **Hybrid Architecture**: Fast path for known intents (saves rate limits), slow path for reasoning
- **One Agent = One Notebook**: No multi-brain routing. NotebookLM is a reading assistant
- **Stateless Chat**: conversation_id=None. smolagents manages ReAct history
- **Pipeline Pattern**: Declarative chaining without LLM calls
- **Error → string conversion**: Tools return errors as strings so CodeAgent can self-correct

## Error Handling

| Error | Recovery |
|-------|----------|
| ChatError (rate limit) | Exponential backoff |
| NetworkError (timeout) | Retry with backoff |
| ValueError (auth expired) | Auto refresh + retry |

## License

Apache-2.0
