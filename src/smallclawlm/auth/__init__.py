"""Authentication for SmallClawLM.

Handles Google OAuth for NotebookLM access via notebooklm-py.
Caches auth tokens in memory to avoid disk reads on every API call.

Two auth sources (in priority order):
  1. SmallClawLM's own auth file (~/.smallclawlm/auth.json)
  2. notebooklm-py's Playwright storage_state.json (fallback)

AuthTokens is async — from_storage() is a coroutine. All public functions
handle the sync/async bridge automatically.
"""

import asyncio
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Path.home() / ".smallclawlm" / "auth.json"

# Singleton cache — avoid reading from disk on every API call
_AUTH_CACHE = None
_CACHE_LOCK = threading.Lock()


async def get_auth(force_refresh: bool = False):
    """Get authenticated tokens for NotebookLM.

    Uses notebooklm-py's AuthTokens.from_storage() which reads the
    Playwright storage_state.json. Results are cached in memory.

    Args:
        force_refresh: If True, invalidate cache and re-read from disk.

    Returns:
        AuthTokens ready for use with NotebookLMClient.

    Raises:
        RuntimeError: If no authentication is found.
    """
    global _AUTH_CACHE

    if _AUTH_CACHE is not None and not force_refresh:
        return _AUTH_CACHE

    with _CACHE_LOCK:
        # Double-check after acquiring lock
        if _AUTH_CACHE is not None and not force_refresh:
            return _AUTH_CACHE

        # Primary: SmallClawLM's own auth file
        if DEFAULT_STORAGE.exists():
            try:
                from notebooklm.auth import AuthTokens
                tokens = await AuthTokens.from_storage(DEFAULT_STORAGE)
                _AUTH_CACHE = tokens
                logger.info("Loaded auth from SmallClawLM storage")
                return tokens
            except Exception as e:
                logger.warning(f"Failed to load SmallClawLM auth: {e}")

        # Fallback: notebooklm-py's default storage
        try:
            from notebooklm.auth import AuthTokens
            tokens = await AuthTokens.from_storage()  # Uses default path
            _AUTH_CACHE = tokens
            logger.info("Loaded auth from notebooklm-py storage")
            return tokens
        except Exception as e:
            raise RuntimeError(
                "No NotebookLM authentication found. "
                "Run: smallclaw login  (or: notebooklm login)"
            ) from e


def get_auth_sync(force_refresh: bool = False):
    """Synchronous wrapper for get_auth().

    Handles the case where we're already inside a running event loop
    (e.g., inside smolagents CodeAgent) by running in a separate thread.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(get_auth(force_refresh))).result()
    else:
        return asyncio.run(get_auth(force_refresh))


def clear_cache() -> None:
    """Clear the in-memory auth cache. Forces next call to re-read from disk."""
    global _AUTH_CACHE
    with _CACHE_LOCK:
        _AUTH_CACHE = None


def ensure_authenticated() -> None:
    """Check if the user is authenticated, raising if not."""
    get_auth_sync()
