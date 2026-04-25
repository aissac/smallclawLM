"""SmallClawLM CLI — Hybrid Fast/Slow Path routing.

Fast path: Direct Pipeline calls for known intents (podcast, report, etc.)
Slow path: CodeAgent with SmolLM3 executor + NotebookLM knowledge tools

Model backends:
  smollm (default): Local SmolLM3-3B via llama-cpp-python (zero external tokens)
  nlm:              Google NotebookLM chat API (requires auth, uses Gemini)

Commands:
  smallclaw run "task"       # Auto-routes: fast path for tool intents, slow for reasoning
  smallclaw agent            # Interactive slow-path session
  smallclaw pipe              # Pipeline: chain operations declaratively
  smallclaw research "topic" # Shortcut: deep research
  smallclaw podcast          # Shortcut: generate podcast
  smallclaw report           # Shortcut: generate report
  smallclaw list              # List notebooks/sources
  smallclaw login             # Authenticate with Google
"""

import asyncio
import click
import logging
import sys

from smallclawlm.router import route, Path as RoutePath
from smallclawlm.notebook_router import NotebookRouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context, handling running event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    else:
        return asyncio.run(coro)



def _resolve_notebook(notebook_id: str | None, query: str | None = None) -> str | None:
    """Auto-resolve notebook_id using NotebookRouter when not provided.

    When --notebook-id / -n is not given, the router picks the best notebook
    based on the query topic using Jaccard similarity + source richness + recency.
    """
    if notebook_id:
        return notebook_id
    if not query:
        return None
    try:
        router = NotebookRouter()
        result = router.route_sync(query)
        click.echo(f"Auto-selected notebook: {result.title} (score={result.score:.2f}, {result.match_level})", err=True)
        if result.created_new:
            click.echo(f"Created new notebook: {result.title} ({result.notebook_id})", err=True)
        return result.notebook_id
    except Exception as e:
        click.echo(f"Warning: notebook routing failed ({e}), using first available", err=True)
        return None


# ─── Fast Path: Direct Pipeline Execution ───

async def _fast_path(intent: str, params: dict, notebook_id: str | None):
    """Execute a known intent directly via Pipeline — no LLM call."""
    from smallclawlm.extensions.pipeline import Pipeline

    pipe = Pipeline(notebook_id=notebook_id)

    if intent == "deep_research":
        query = params.get("query", "general research")
        mode = params.get("mode", "deep")
        pipe.research(query, mode=mode)
        result = await pipe.execute_async()
        return result

    elif intent == "generate_podcast":
        pipe.generate("podcast")
        result = await pipe.execute_async()
        return result

    elif intent == "generate_video":
        pipe.generate("video")
        result = await pipe.execute_async()
        return result

    elif intent == "generate_quiz":
        pipe.generate("quiz")
        result = await pipe.execute_async()
        return result

    elif intent == "generate_mind_map":
        pipe.generate("mindmap")
        result = await pipe.execute_async()
        return result

    elif intent == "generate_report":
        pipe.generate("report")
        result = await pipe.execute_async()
        return result

    elif intent == "list_sources":
        from smallclawlm.auth import get_auth
        from notebooklm import NotebookLMClient
        auth = await get_auth()
        async with NotebookLMClient(auth) as client:
            if not notebook_id:
                notebooks = await client.notebooks.list()
                if not notebooks:
                    return "No notebooks found."
                notebook_id = notebooks[0].id
            sources = await client.sources.list(notebook_id)
            if not sources:
                return f"No sources in notebook {notebook_id}."
            lines = [f"Sources in notebook {notebook_id}:"]
            for s in sources:
                name = getattr(s, 'title', None) or getattr(s, 'filename', str(s.id))
                lines.append(f"  - {name} ({s.id})")
            return "\n".join(lines)

    elif intent == "add_source":
        url = params.get("url", "")
        if not url:
            return "Error: no URL provided. Usage: smallclaw run 'add source <url>'"
        pipe.add_source(url)
        result = await pipe.execute_async()
        return result

    elif intent == "create_notebook":
        from smallclawlm.auth import get_auth
        from notebooklm import NotebookLMClient
        auth = await get_auth()
        async with NotebookLMClient(auth) as client:
            title = params.get("title", "SmallClawLM Notebook")
            nb = await client.notebooks.create(title)
            return f"Created notebook '{title}' (id: {nb.id})"

    else:
        return f"Unknown intent: {intent}"


async def _slow_nlm_path(task: str, notebook_id: str | None, max_steps: int = 10):
    """Execute a task using NLMAgent with NLMModel brain — for complex reasoning."""
    from smallclawlm import NLMAgent

    agent = NLMAgent(
        notebook_id=notebook_id,
        tools="all",
        max_steps=max_steps,
    )
    return agent.run(task)


def _slow_path(task: str, notebook_id: str | None, max_steps: int = 10, model_backend: str = "smollm"):
    """Execute a task using NLMAgent — SmolLM3 (local) or NLM (cloud)."""
    from smallclawlm import NLMAgent

    agent = NLMAgent(
        notebook_id=notebook_id,
        tools="all",
        max_steps=max_steps,
    )
    return agent.run(task)


# ─── CLI Commands ───

@click.group()
@click.version_option(version="0.4.0", prog_name="smallclaw")
def cli():
    """SmallClawLM - Zero-token AI agent powered by SmolLM3 + NotebookLM."""
    pass


@cli.command()
def login():
    """Authenticate with Google NotebookLM (opens browser)."""
    click.echo("Opening browser for Google authentication...")
    try:
        from notebooklm.cli.session import run_login
        run_login()
        click.echo("Authentication successful!")
    except Exception as e:
        click.echo(f"Authentication failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
def auth_check(notebook_id):
    """Check if current authentication is valid."""
    from smallclawlm.auth import get_auth
    try:
        _run_async(get_auth())
        click.echo("Authentication is valid!")
    except RuntimeError as e:
        click.echo(f"{e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("task")
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--max-steps", "-m", default=10, help="Max agent steps (slow path only)")
@click.option("--model", "model_backend", type=click.Choice(["smollm", "nlm"]), default="smollm",
              help="Model backend: smollm (local SmolLM3) or nlm (NotebookLM cloud)")
@click.option("--force-slow", is_flag=True, help="Force slow path (agent) even for known intents")
@click.option("--force-fast", is_flag=True, help="Force fast path (pipeline) even for conversational input")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(task, notebook_id, max_steps, model_backend, force_slow, force_fast, verbose):
    """Run a task with automatic fast/slow path routing.

    By default, the router determines whether to use fast path (direct tool call)
    or slow path (agent reasoning). Use --force-slow or --force-fast to override.

    Model backends:
      smollm (default): Local SmolLM3-3B, zero external tokens
      nlm: NotebookLM chat API (requires auth)
    """
    if force_fast and force_slow:
        click.echo("Error: cannot use both --force-fast and --force-slow", err=True)
        sys.exit(1)

    # Auto-resolve notebook if not specified
    if not notebook_id:
        notebook_id = _resolve_notebook(None, task)

    # Route the task
    result = route(task)

    if verbose:
        click.echo(f"Route: {result.path.value} path → {result.intent} (confidence: {result.confidence:.0%})")
        click.echo(f"Model: {model_backend}")

    # Override if requested
    if force_slow:
        result = result.__class__(path=RoutePath.SLOW, intent="ask", confidence=0.5, params={"query": task})
    elif force_fast:
        if result.intent in ("deep_research", "generate_podcast", "generate_video",
                              "generate_quiz", "generate_mind_map", "generate_report",
                              "list_sources", "add_source", "create_notebook"):
            result = result.__class__(path=RoutePath.FAST, intent=result.intent, confidence=1.0, params=result.params)
        else:
            click.echo("Error: no fast path available for this intent", err=True)
            sys.exit(1)

    # Execute
    if result.path == RoutePath.FAST:
        if not notebook_id:
            notebook_id = _resolve_notebook(None, task)
        if verbose:
            click.echo(f"Fast path: {result.intent}")
        output = _run_async(_fast_path(result.intent, result.params, notebook_id))
    else:
        if verbose:
            click.echo(f"Slow path: {model_backend} agent")
        output = _slow_path(task, notebook_id, max_steps, model_backend)

    click.echo(output)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--max-steps", "-m", default=10, help="Max agent steps")
@click.option("--model", "model_backend", type=click.Choice(["smollm", "nlm"]), default="smollm",
              help="Model backend: smollm (local) or nlm (cloud)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def agent(notebook_id, max_steps, model_backend, verbose):
    """Start an interactive agent session (always slow path).

    Model backends:
      smollm (default): Local SmolLM3-3B, zero external tokens
      nlm: NotebookLM chat API (requires auth)
    """
    from smallclawlm import NLMAgent

    if not notebook_id:
        notebook_id = _resolve_notebook(None, "interactive session")

    nlm_agent = NLMAgent(
        notebook_id=notebook_id,
        tools="all",
        max_steps=max_steps,
    )

    model_label = "SmolLM3 (local)" if model_backend == "smollm" else "NotebookLM (cloud)"
    click.echo(f"SmallClawLM Agent (slow path — {model_label})")
    click.echo("Type your tasks, Ctrl+C to exit")
    click.echo("=" * 50)

    while True:
        try:
            task = click.prompt("Task")
            if task.strip().lower() in ("exit", "quit", "q"):
                break
            result = nlm_agent.run(task)
            click.echo(result)
        except KeyboardInterrupt:
            click.echo("\nBye!")
            break


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--add-source", "-s", multiple=True, help="Add URL as source (can repeat)")
@click.option("--generate", "-g", type=click.Choice(["podcast", "video", "quiz", "mindmap", "report"]),
              help="Generate an artifact")
@click.option("--ask", "-a", "question", help="Ask a question about sources")
@click.option("--research", "-r", help="Run deep research on a topic")
def pipe(notebook_id, add_source, generate, question, research):
    """Pipeline: chain operations without an agent (fast path)."""
    from smallclawlm.extensions.pipeline import Pipeline, ArtifactType

    pipe = Pipeline(notebook_id=notebook_id)

    for url in add_source:
        pipe.add_source(url)

    if research:
        pipe.research(research)

    if question:
        pipe.ask(question)

    if generate:
        pipe.generate(ArtifactType(generate))

    if not add_source and not question and not research and not generate:
        click.echo("No operations specified. Use --add-source, --ask, --research, or --generate.")
        return

    result = pipe.execute()
    click.echo(result)


# ─── Shortcut Commands ───

@cli.command()
@click.argument("query")
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--mode", type=click.Choice(["fast", "deep"]), default="deep", help="Research depth")
def research(query, notebook_id, mode):
    """Run deep research on a topic (fast path shortcut)."""
    if not notebook_id:
        notebook_id = _resolve_notebook(None, query)
    output = _run_async(_fast_path("deep_research", {"query": query, "mode": mode}, notebook_id))
    click.echo(output)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--instructions", "-i", default=None, help="Custom instructions for the podcast")
def podcast(notebook_id, instructions):
    """Generate a podcast from notebook sources (fast path shortcut)."""
    if not notebook_id:
        notebook_id = _resolve_notebook(None, "podcast generation")
    output = _run_async(_fast_path("generate_podcast", {"instructions": instructions}, notebook_id))
    click.echo(output)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
def report(notebook_id):
    """Generate a report from notebook sources (fast path shortcut)."""
    if not notebook_id:
        notebook_id = _resolve_notebook(None, "report generation")
    output = _run_async(_fast_path("generate_report", {}, notebook_id))
    click.echo(output)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
def quiz(notebook_id):
    """Generate a quiz from notebook sources (fast path shortcut)."""
    if not notebook_id:
        notebook_id = _resolve_notebook(None, "quiz generation")
    output = _run_async(_fast_path("generate_quiz", {}, notebook_id))
    click.echo(output)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
def mindmap(notebook_id):
    """Generate a mind map from notebook sources (fast path shortcut)."""
    if not notebook_id:
        notebook_id = _resolve_notebook(None, "mind map generation")
    output = _run_async(_fast_path("generate_mind_map", {}, notebook_id))
    click.echo(output)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
def list_sources(notebook_id):
    """List sources in a notebook."""
    if not notebook_id:
        notebook_id = _resolve_notebook(None, "list sources")
    output = _run_async(_fast_path("list_sources", {}, notebook_id))
    click.echo(output)



@cli.command()
@click.option("--token", "-t", envvar="TELEGRAM_BOT_TOKEN", help="Telegram bot token")
@click.option("--max-steps", "-m", default=10, help="Max agent steps")
@click.option("--n-threads", default=4, help="SmolLM3 threads (optimal: 4)")
@click.option("--n-ctx", default=2048, help="SmolLM3 context window size")
def serve(token, max_steps, n_threads, n_ctx):
    """Start a Telegram bot gateway.

    Set TELEGRAM_BOT_TOKEN env var or pass --token.
    The bot auto-selects notebooks based on message content.
    """
    if not token:
        click.echo("Error: TELEGRAM_BOT_TOKEN required. Set env var or pass --token.", err=True)
        sys.exit(1)

    try:
        from smallclawlm.gateways.telegram import TelegramGateway
    except ImportError:
        click.echo("Error: python-telegram-bot not installed. Install with: pip install smallclawlm[telegram]", err=True)
        sys.exit(1)

    gateway = TelegramGateway(
        token=token,
        max_steps=max_steps,
        n_threads=n_threads,
        n_ctx=n_ctx,
    )
    click.echo("SmallClawLM Telegram gateway starting...")
    gateway.run(block=True)


if __name__ == "__main__":
    cli()
