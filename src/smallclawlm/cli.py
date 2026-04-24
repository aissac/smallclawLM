"""SmallClawLM CLI — Command-line interface."""

import click
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="smallclaw")
def cli():
    """🦞 SmallClawLM — Zero-token AI agent powered by Google NotebookLM."""
    pass


@cli.command()
def login():
    """Authenticate with Google NotebookLM (opens browser)."""
    click.echo("🔐 Opening browser for Google authentication...")
    try:
        from notebooklm.cli.session import run_login
        run_login()
        click.echo("✅ Authentication successful!")
    except Exception as e:
        click.echo(f"❌ Authentication failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def auth_check():
    """Check if current authentication is valid."""
    from smallclawlm.auth import get_auth
    try:
        asyncio.run(get_auth())
        click.echo("✅ Authentication is valid!")
    except RuntimeError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--notebook-title", "-t", default="SmallClawLM Session", help="Title for auto-created notebook")
@click.option("--max-steps", "-m", default=20, help="Maximum agent reasoning steps")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def agent(notebook_id, notebook_title, max_steps, verbose):
    """Start an interactive SmallClawLM agent session."""
    from smallclawlm import NLMAgent

    nlm_agent = NLMAgent(
        notebook_id=notebook_id,
        notebook_title=notebook_title,
        max_steps=max_steps,
        verbosity_level=2 if verbose else 1,
    )

    click.echo("🦞 SmallClawLM Agent (type your tasks, Ctrl+C to exit)")
    click.echo("=" * 50)

    while True:
        try:
            task = click.prompt("\nTask")
            if task.strip().lower() in ("exit", "quit", "q"):
                break
            result = nlm_agent.run(task)
            click.echo(f"\n🦞 {result}")
        except KeyboardInterrupt:
            click.echo("\n👋 Bye!")
            break


@cli.command()
@click.argument("task")
@click.option("--notebook-id", "-n", default=None, help="NotebookLM notebook ID")
@click.option("--max-steps", "-m", default=20, help="Maximum agent steps")
def run(task, notebook_id, max_steps):
    """Run a single task and exit."""
    from smallclawlm import NLMAgent

    nlm_agent = NLMAgent(
        notebook_id=notebook_id,
        max_steps=max_steps,
    )
    result = nlm_agent.run(task)
    click.echo(result)


@cli.command()
@click.option("--research", "-r", help="Research query")
@click.option("--generate", "-g", type=click.Choice(["podcast", "video", "quiz", "flashcards", "mindmap", "report"]), help="Artifact to generate")
@click.option("--download", "-d", default=None, help="Download output to path")
@click.option("--add-source", "-s", multiple=True, help="Add source URL (repeatable)")
@click.option("--notebook-id", "-n", default=None, help="Notebook ID")
def pipe(research, generate, download, add_source, notebook_id):
    """Run a pipeline: research → generate → download."""
    click.echo("🦞 Running pipeline...")

    async def _run():
        from smallclawlm.auth import get_auth
        from notebooklm import NotebookLMClient

        auth = await get_auth()
        async with NotebookLMClient(auth) as client:
            # Get or create notebook
            if notebook_id:
                nb_id = notebook_id
            else:
                nb = await client.notebooks.create("Pipeline Session")
                nb_id = nb.id
                click.echo(f"📝 Created notebook: {nb_id}")

            # Add sources
            for url in add_source:
                src = await client.sources.add_url(nb_id, url)
                click.echo(f"📎 Added source: {src.id}")

            # Research
            if research:
                click.echo(f"🔍 Researching: {research}")
                r = await client.research.start(
                    notebook_id=nb_id, query=research, source="web", mode="deep"
                )
                poll = await client.research.poll(nb_id)
                click.echo(f"✅ Research complete")

            # Generate
            if generate:
                click.echo(f"🎨 Generating {generate}...")
                gen_map = {
                    "podcast": client.artifacts.generate_audio,
                    "video": client.artifacts.generate_video,
                    "quiz": client.artifacts.generate_quiz,
                    "flashcards": client.artifacts.generate_flashcards,
                    "mindmap": client.artifacts.generate_mind_map,
                    "report": client.artifacts.generate_report,
                }
                gen_fn = gen_map[generate]
                result = await gen_fn(nb_id)
                click.echo(f"✅ Generation started: {getattr(result, 'task_id', 'done')}")

                # Wait for completion
                click.echo("⏳ Waiting for generation to complete...")
                # TODO: implement wait + download

            click.echo("🦞 Pipeline complete!")

    try:
        asyncio.run(_run())
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--mode", "-m", type=click.Choice(["fast", "deep"]), default="deep", help="Research mode")
@click.option("--notebook-id", "-n", default=None, help="Notebook ID")
def research(query, mode, notebook_id):
    """Run a research query on NotebookLM."""
    async def _run():
        from smallclawlm.auth import get_auth
        from notebooklm import NotebookLMClient

        auth = await get_auth()
        async with NotebookLMClient(auth) as client:
            nb_id = notebook_id
            if not nb_id:
                notebooks = await client.notebooks.list()
                if notebooks:
                    nb_id = notebooks[0].id
                else:
                    nb = await client.notebooks.create("Research")
                    nb_id = nb.id

            click.echo(f"🔍 Running {mode} research: {query}")
            r = await client.research.start(
                notebook_id=nb_id, query=query, source="web", mode=mode
            )
            poll = await client.research.poll(nb_id)
            if poll.get("report"):
                click.echo(poll["report"])
            else:
                click.echo("✅ Research complete. Use 'smallclaw agent' to chat with results.")

    try:
        asyncio.run(_run())
    except Exception as e:
        click.echo(f"❌ Research failed: {e}", err=True)
        sys.exit(1)


@cli.group()
def serve():
    """Start SmallClawLM as a bot on a messaging platform."""
    pass


@serve.command()
@click.option("--token", envvar="TELEGRAM_BOT_TOKEN", required=True, help="Telegram bot token")
def telegram(token):
    """Start SmallClawLM as a Telegram bot."""
    click.echo("🦞 Starting Telegram bot...")
    click.echo("⚠️ Telegram gateway is coming soon! Track progress at github.com/aissac/smallclawLM")


@serve.command()
@click.option("--token", envvar="DISCORD_BOT_TOKEN", required=True, help="Discord bot token")
def discord(token):
    """Start SmallClawLM as a Discord bot."""
    click.echo("🦞 Starting Discord bot...")
    click.echo("⚠️ Discord gateway is coming soon!")


@serve.command()
@click.option("--port", "-p", default=8000, help="Web server port")
def web(port):
    """Start SmallClawLM as a REST API."""
    click.echo("🦞 Starting web API...")
    click.echo("⚠️ Web gateway is coming soon!")


if __name__ == "__main__":
    cli()
