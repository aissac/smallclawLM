"""Authentication for SmallClawLM.

Handles Google OAuth for NotebookLM access, including:
- Browser-based login (via notebooklm-py)
- Token storage and retrieval
- Auto-refresh for expired sessions
- Caching to avoid repeated disk reads
"""

import asyncio
import json
import logging
from pathlib import Path

from notebooklm.auth import AuthTokens, extract_cookies_from_storage, fetch_tokens
from notebooklm.paths import get_storage_path

logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Path.home() / ".smallclawlm" / "auth.json"

# Singleton cache — avoid reading from disk on every API call
_AUTH_CACHE: AuthTokens | None = None


async def get_auth(force_refresh: bool = False) -> AuthTokens:
    """Get authenticated tokens for NotebookLM.

    Tries SmallClawLM's own auth first, then falls back to notebooklm-py's
    stored credentials. Results are cached in memory.

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

    # Try SmallClawLM's own auth first
    if DEFAULT_STORAGE.exists():
        try:
            tokens = await _load_auth(DEFAULT_STORAGE)
            _AUTH_CACHE = tokens
            return tokens
        except Exception as e:
            logger.warning(f"Failed to load SmallClawLM auth: {e}")

    # Fall back to notebooklm-py's stored auth
    try:
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


async def _load_auth(path: Path) -> AuthTokens:
    """Load auth tokens from a JSON file."""
    data = json.loads(path.read_text())
    cookies = extract_cookies_from_storage(data)
    csrf, session_id = await fetch_tokens(cookies)
    return AuthTokens(cookies=cookies, csrf_token=csrf, session_id=session_id)


def ensure_authenticated() -> None:
    """Check if the user is authenticated, raise if not."""
    try:
        asyncio.run(get_auth())
    except RuntimeError:
        raise


def clear_cache() -> None:
    """Clear the in-memory auth cache. Forces next call to re-read from disk."""
    global _AUTH_CACHE
    _AUTH_CACHE = None
