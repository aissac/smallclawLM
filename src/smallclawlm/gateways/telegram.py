"""SmallClawLM Telegram Gateway — Bot polling + message routing.

Runs a Telegram bot that receives messages, routes them through
NotebookRouter to select the right notebook, processes via NLMAgent,
and sends responses back.

Usage:
  smallclaw serve --platform telegram --token <BOT_TOKEN>

Or programmatically:
  from smallclawlm.gateways.telegram import TelegramGateway
  gateway = TelegramGateway(token="123:ABC")
  gateway.run()

Features:
  - Long polling (no webhook setup needed)
  - Auto notebook selection via NotebookRouter
  - Per-chat notebook affinity (remembers which notebook each chat uses)
  - /notebook command to see/switch current notebook
  - /new command to start a fresh notebook
  - /notebooks command to list all notebooks
  - Rich formatting: bold, italic, code blocks
"""

import asyncio
import logging
import re
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from smallclawlm.gateways.common import GatewayMessage, GatewayResponse, MessageType, format_route_info
from smallclawlm.notebook_router import NotebookRouter, RouteResult
from smallclawlm.router import route, Path as RoutePath
from smallclawlm.nlm_agent import NLMAgent

logger = logging.getLogger(__name__)

# Telegram message limit is 4096 chars
MAX_RESPONSE_LEN = 4000
TRUNCATE_SUFFIX = "\n\n... (response truncated)"


class TelegramGateway:
    """Telegram bot gateway for SmallClawLM.

    Each chat gets its own notebook selection. The router picks the best
    notebook based on the first message, then sticks with it until the
    user switches with /notebook or /new.
    """

    def __init__(
        self,
        token: str,
        max_steps: int = 10,
    ):
        self.token = token
        self.max_steps = max_steps

        # Per-chat state: chat_id -> {"notebook_id": str, "agent": NLMAgent}
        self._chat_state: dict[int, dict] = {}

        # NotebookRouter for auto-selection
        self._router = NotebookRouter()

        # Build the Telegram application
        self._app = Application.builder().token(token).build()
        self._register_handlers()

    def _register_handlers(self):
        """Register command and message handlers."""
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("notebooks", self._cmd_notebooks))
        self._app.add_handler(CommandHandler("notebook", self._cmd_notebook))
        self._app.add_handler(CommandHandler("new", self._cmd_new))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

    # ─── Commands ───

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start — welcome message."""
        chat_id = update.effective_chat.id
        await update.message.reply_text(
            "*SmallClawLM* — Zero-token AI agent powered by NotebookLM\n\n"
            "Just send me a task and I\'ll auto-select the right notebook.\n\n"
            "Commands:\n"
            "/notebooks — List all notebooks\n"
            "/notebook — Show current notebook\n"
            "/new — Create a fresh notebook\n"
            "/help — Show this message",
            parse_mode="Markdown",
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help."""
        await self._cmd_start(update, context)

    async def _cmd_notebooks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notebooks — list all available notebooks."""
        try:
            await self._router.refresh()
            if not self._router._metadata:
                await update.message.reply_text("No notebooks found. Send me a message and I\'ll create one!")
                return

            lines = ["*Notebooks:*\n"]
            for nb_id, meta in self._router._metadata.items():
                chat_nb = self._chat_state.get(update.effective_chat.id, {}).get("notebook_id")
                marker = " ← current" if nb_id == chat_nb else ""
                lines.append(f"• {meta.title} ({meta.source_count} sources){marker}")

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"Error listing notebooks: {e}")

    async def _cmd_notebook(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /notebook — show current notebook or switch to one."""
        chat_id = update.effective_chat.id
        state = self._chat_state.get(chat_id, {})

        # If args provided, switch notebook
        if context.args:
            query = " ".join(context.args)
            try:
                result = self._router.route_sync(query)
                state["notebook_id"] = result.notebook_id
                self._chat_state[chat_id] = state
                await update.message.reply_text(
                    f"Switched to: *{result.title}* (score: {result.score:.2f}, {result.match_level})",
                    parse_mode="Markdown",
                )
                # Reset agent for new notebook
                if "agent" in state:
                    del state["agent"]
                return
            except Exception as e:
                await update.message.reply_text(f"Error switching notebook: {e}")
                return

        # No args — show current
        nb_id = state.get("notebook_id")
        if nb_id and self._router._metadata:
            meta = self._router._metadata.get(nb_id)
            title = meta.title if meta else nb_id[:16]
            await update.message.reply_text(f"Current notebook: *{title}*", parse_mode="Markdown")
        else:
            await update.message.reply_text("No notebook selected. Send me a message and I\'ll auto-select one!")

    async def _cmd_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new — create a fresh notebook for this chat."""
        chat_id = update.effective_chat.id
        title = " ".join(context.args) if context.args else f"SmallClawLM Chat {chat_id}"

        try:
            from smallclawlm.auth import get_auth
            from notebooklm import NotebookLMClient
            auth = await get_auth()
            async with NotebookLMClient(auth) as client:
                nb = await client.notebooks.create(title)

            state = self._chat_state.get(chat_id, {})
            state["notebook_id"] = nb.id
            # Reset agent
            if "agent" in state:
                del state["agent"]
            self._chat_state[chat_id] = state

            await self._router.refresh(force=True)
            await update.message.reply_text(
                f"Created new notebook: *{title}*\nID: `{nb.id}`",
                parse_mode="Markdown",
            )
        except Exception as e:
            await update.message.reply_text(f"Error creating notebook: {e}")

    # ─── Message Handler ───

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages — the main agent interaction."""
        chat_id = update.effective_chat.id
        user_text = update.message.text

        if not user_text or not user_text.strip():
            return

        # Send "thinking" indicator
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # Resolve notebook for this chat
        state = self._chat_state.get(chat_id, {})
        notebook_id = state.get("notebook_id")

        if not notebook_id:
            try:
                result = self._router.route_sync(user_text)
                notebook_id = result.notebook_id
                state["notebook_id"] = notebook_id
                route_info = f"Auto-selected: *{result.title}* ({result.match_level})"
                if result.created_new:
                    route_info = f"Created new notebook: *{result.title}*"
                await update.message.reply_text(route_info, parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text(f"Error selecting notebook: {e}")
                return

        self._chat_state[chat_id] = state

        # Get or create agent for this chat
        agent = state.get("agent")
        if agent is None:
            try:
                agent = NLMAgent(
                    notebook_id=notebook_id,
                    tools="all",
                    max_steps=self.max_steps,
                )
                state["agent"] = agent
                self._chat_state[chat_id] = state
            except Exception as e:
                await update.message.reply_text(f"Error creating agent: {e}")
                return

        # Route: fast path vs slow path
        route_result = route(user_text)

        try:
            if route_result.path == RoutePath.FAST:
                response_text = await self._fast_path(route_result, notebook_id)
            else:
                response_text = await self._slow_path(agent, user_text)

            # Truncate if needed
            if len(response_text) > MAX_RESPONSE_LEN:
                response_text = response_text[:MAX_RESPONSE_LEN] + TRUNCATE_SUFFIX

            # Send response
            await update.message.reply_text(response_text, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await update.message.reply_text(f"Error: {e}")

    async def _fast_path(self, route_result, notebook_id: str) -> str:
        """Execute a fast-path intent directly."""
        try:
            from smallclawlm.auth import get_auth
            from notebooklm import NotebookLMClient

            auth = await get_auth()
            async with NotebookLMClient(auth) as client:
                intent = route_result.intent
                params = route_result.params

                if intent == "deep_research":
                    query = params.get("query", "general research")
                    await client.research.start(notebook_id=notebook_id, query=query, source="web", mode="deep")
                    return f"Deep research on \'{query}\' started. Results will be added to the notebook."

                elif intent == "generate_podcast":
                    await client.artifacts.generate_podcast(notebook_id=notebook_id)
                    return "Podcast generation started. Check back in a few minutes."

                elif intent == "generate_report":
                    await client.artifacts.generate_report(notebook_id=notebook_id)
                    return "Report generation started. Check back in a few minutes."

                elif intent == "generate_quiz":
                    await client.artifacts.generate_quiz(notebook_id=notebook_id)
                    return "Quiz generation started. Check back in a few minutes."

                elif intent == "generate_mind_map":
                    await client.artifacts.generate_mindmap(notebook_id=notebook_id)
                    return "Mind map generation started."

                elif intent == "generate_video":
                    await client.artifacts.generate_video(notebook_id=notebook_id)
                    return "Video generation started."

                elif intent == "list_sources":
                    sources = await client.sources.list(notebook_id)
                    if not sources:
                        return "No sources in this notebook."
                    lines = [f"Sources ({len(sources)}):"]
                    for s in sources[:20]:
                        name = getattr(s, "title", None) or getattr(s, "filename", str(s.id))
                        lines.append(f"  • {name}")
                    if len(sources) > 20:
                        lines.append(f"  ... and {len(sources) - 20} more")
                    return "\n".join(lines)

                else:
                    return f"Unknown intent: {intent}"

        except Exception as e:
            return f"Error: {e}"

    async def _slow_path(self, agent: NLMAgent, task: str) -> str:
        """Execute a slow-path task through the NLMAgent."""
        # NLMAgent.run() is sync, so run it in a thread
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, agent.run, task)
        return str(result) if result else "No response from agent."

    # ─── Run ───

    def run(self, block: bool = True):
        """Start the Telegram bot (long polling).

        Args:
            block: If True, blocks until interrupted. If False, starts in background.
        """
        logger.info("Starting SmallClawLM Telegram gateway...")
        if block:
            self._app.run_polling(allowed_updates=Update.ALL_TYPES)
        else:
            # For integration into existing event loops
            return self._app

    async def run_async(self):
        """Start the bot asynchronously."""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("Telegram gateway running (async)")

    async def stop(self):
        """Stop the bot gracefully."""
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("Telegram gateway stopped")
