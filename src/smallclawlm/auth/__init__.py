"""Authentication for SmallClawLM.

Handles Google OAuth for NotebookLM access via notebooklm-py.
Caches auth tokens in memory to avoid disk reads on every API call.
"""

import asyncio
import json
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

    Tries SmallClawLM's own auth first, then falls back to notebooklm-py's
    stored credentials (Playwright storage_state.json). Results are cached.

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

        # Try SmallClawLM's own auth first
        if DEFAULT_STORAGE.exists():
            try:
                tokens = await _load_auth(DEFAULT_STORAGE)
                _AUTH_CACHE = tokens
                return tokens
            except Exception as e:
                logger.warning(f"Failed to load SmallClawLM auth: {e}")

        # Fall back to notebooklm-py's stored auth (Playwright storage_state.json)
        try:
            from notebooklm.auth import AuthTokens, extract_cookies_from_storage, fetch_tokens
            from notebooklm.paths import get_storage_path

            storage = get_storage_path()
            storage_state = json.loads(storage.read_text())
            cookies = extract_cookies_from_storage(storage_state)
            csrf, session_id = await fetch_tokens(cookies)
            tokens = AuthTokens(cookies=cookies, csrf_token=csrf, session_id=session_id)
            _AUTH_CACHE = tokens
            return tokens
        except Exception as e:
            raise RuntimeError(
                "No NotebookLM authentication found. Run: smallclaw login"
            ) from e


async def _load_auth(path: Path):
    """Load auth tokens from a JSON file."""
    from notebooklm.auth import AuthTokens, extract_cookies_from_storage, fetch_tokens

    data = json.loads(path.read_text())
    cookies = extract_cookies_from_storage(data)
    csrf, session_id = await fetch_tokens(cookies)
    return AuthTokens(cookies=cookies, csrf_token=csrf, session_id=session_id)


def clear_cache() -> None:
    """Clear the in-memory auth cache. Forces next call to re-read from disk."""
    global _AUTH_CACHE
    with _CACHE_LOCK:
        _AUTH_CACHE = None


def ensure_authenticated() -> None:
    """Check if the user is authenticated, raising if not."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an event loop — schedule it
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(lambda: asyncio.run(get_auth())).result()
    else:
        asyncio.run(get_auth())
