# 🦞 SmallClawLM

**Zero-token AI agent powered by Google NotebookLM.**

SmallClawLM is an open-source agent framework that uses Google NotebookLM's built-in Gemini as its "brain" — no external LLM API keys needed. Built on [smolagents](https://github.com/huggingface/smolagents) + [notebooklm-py](https://github.com/teng-lin/notebooklm-py).

## Why?

NotebookLM provides:
- **Deep research** with web browsing
- **Source-grounded Q&A** chat
- **Audio, video, quiz, flashcard** generation
- **Mind maps, infographics, slide decks**
- **Reports and data tables**

...all powered by Google's Gemini, for free. SmallClawLM wraps this into an autonomous agent that can plan, research, generate, and iterate — without spending a single token on external LLM APIs.

## How It Works

```
User goal → smolagents CodeAgent
              ↓
         NLMModel (routes to NotebookLM chat API — Gemini inside Google)
              ↓
         NLMTools (research, generate, sources, etc.)
              ↓
         Agent decides next step using NotebookLM's own intelligence
              ↓
         Result back to user
```

The agent **thinks** using NotebookLM's chat, **acts** using NotebookLM's APIs, and **generates** using NotebookLM's artifact engine. The entire intelligence loop runs on Google's infrastructure — zero external tokens.

## Quick Start

```bash
pip install smallclawlm

# Authenticate with Google (one-time, opens browser)
smallclaw login

# Interactive agent
smallclaw agent

# One-shot query
smallclaw run "Research fusion energy advances and create a podcast"

# Pipeline: research → generate → download
smallclaw pipe --research "quantum computing" --generate podcast --download ./output.mp3
```

## Architecture

```
smallclawlm/
├── nlm_model.py           # smolagents Model → NotebookLM chat (the brain!)
├── nlm_tools.py            # smolagents Tool wrappers around NotebookLMClient
├── nlm_agent.py            # Pre-configured CodeAgent with NLM as brain
├── auth/                   # Authentication (browser, token refresh)
├── extensions/             # Pipelines, batch, dedup, templates
├── memory.py               # AgentMemory backed by NLM notebooks
├── cli.py                  # CLI: smallclaw agent/run/pipe/research
└── gateways/               # Telegram, Discord, Web adapters
```

## Key Components

### NLMModel (The Brain)

Wraps NotebookLM's chat API as a smolagents `Model`:

```python
from smallclawlm import NLMModel, NLMAgent

model = NLMModel()  # Uses your NotebookLM session
agent = NLMAgent(model=model)
result = agent.run("Analyze these research papers and create a study guide")
```

### NLMTools (The Hands)

| Tool | Description |
|------|-------------|
| `DeepResearchTool` | Run deep web research on a topic |
| `GeneratePodcastTool` | Generate audio overview podcasts |
| `GenerateVideoTool` | Generate video explainers |
| `GenerateQuizTool` | Create quizzes from sources |
| `GenerateFlashcardsTool` | Create study flashcards |
| `GenerateMindMapTool` | Generate visual mind maps |
| `GenerateReportTool` | Generate structured reports |
| `AddSourceTool` | Add URLs, PDFs, YouTube videos |
| `ListSourcesTool` | List all sources in a notebook |
| `CreateNotebookTool` | Create a new notebook |
| `DedupSourcesTool` | Remove duplicate sources |

### Pipelines

Chain operations without an agent:

```bash
# Research → Generate podcast → Download
smallclaw pipe --research "climate tech" --generate podcast --download ./podcast.mp3
```

## License

Apache-2.0 — same as smolagents.

## Acknowledgments

- [smolagents](https://github.com/huggingface/smolagents) by Hugging Face
- [notebooklm-py](https://github.com/teng-lin/notebooklm-py) by Teng Lin
- [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research
