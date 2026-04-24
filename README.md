# SmallClawLM

**Zero-token AI agent powered by Google NotebookLM.**

One agent, one notebook, one specialty. Each SmallClawLM agent owns exactly one NotebookLM notebook - the notebook's sources define its domain expertise. All reasoning runs through Gemini inside NotebookLM. **No external LLM API keys needed.**

## How It Works

Each agent is a smolagents CodeAgent backed by NLMModel (calls chat.ask on its notebook). Tools operate on the same notebook. No multi-brain routing needed.

## Agent Specialties

| Specialty | Tools | Use Case |
|-----------|-------|----------|
| research | deep_research, ask_notebook, add_source, list_sources, create_notebook | Web research and Q&A |
| podcast | ask_notebook, add_source, generate_podcast | Audio overviews |
| quiz | ask_notebook, add_source, generate_quiz | Study aids |
| report | ask_notebook, add_source, generate_report | Structured reports |
| mindmap | ask_notebook, add_source, generate_mind_map | Visual summaries |
| all | Everything + direct_response | General purpose |

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

### CLI

```bash
# Research agent (default)
smallclaw run "What are the latest breakthroughs in fusion energy?"

# Podcast agent
smallclaw run --specialty podcast "Create a podcast about climate change"

# Interactive session
smallclaw agent --specialty research

# Pipeline: add sources, research, generate
smallclaw pipe --add-source "https://arxiv.org/abs/2401.12345" --generate podcast
```

### Python API

```python
from smallclawlm import NLMAgent, create_agent

# Create a research agent
agent = create_agent("research")
result = agent.run("What are the latest fusion energy breakthroughs?")

# Create a podcast agent with a specific notebook
agent = NLMAgent(notebook_id="abc123", tools="podcast")
result = agent.run("Create a podcast about quantum computing")
```

## Architecture Decisions

- **One Agent = One Notebook**: No multi-brain routing. NotebookLM is a reading assistant, not a router.
- **Stateless Chat**: conversation_id=None. Smolagents manages ReAct history. No context drift.
- **ChatMode.CONCISE**: Shorter responses, faster agent loops (~30% less generation time).
- **CodeAgent**: NotebookLM outputs Python code blocks naturally. No forced JSON tool calls.
- **Source-Based Format Teaching**: Format rules uploaded as source docs, not prompts.
- **Fuzzy Parser**: 3-tier fallback for imperfect NLM responses.

## Error Handling

| Error | Recovery |
|-------|----------|
| ChatError (rate limit) | Exponential backoff (2s, 4s, 8s) |
| NetworkError (timeout) | Retry with backoff |
| ValueError (auth expired) | Auto refresh_auth() + retry |

## Known Limitations

- 5-15s per NotebookLM call (vs <1s for direct API)
- notebooklm-py is unofficial - Google may break it
- No structured JSON output - prose + code blocks only
- Undocumented rate limits

## License

Apache-2.0

## Credits

Built on [smolagents](https://github.com/huggingface/smolagents) and [notebooklm-py](https://github.com/nas5f/notebooklm-py).
