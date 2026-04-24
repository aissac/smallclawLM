"""Authentication for SmallClawLM.

Handles Google OAuth for NotebookLM access, including:
- Browser-based login (opens browser for user authentication)
- Token storage and retrieval
- Auto-refresh for expired sessions
"""

import asyncio
import json
import logging
from pathlib import Path

from notebooklm.auth import AuthTokens, extract_cookies_from_storage, fetch_tokens
from notebooklm.paths import get_storage_path

logger = logging.getLogger(__name__)

DEFAULT_STORAGE = Path.home() / ".smallclawlm" / "auth.json"


async def get_auth() -> AuthTokens:
    """Get authenticated tokens for NotebookLM.

    Tries SmallClawLM's own auth first, then falls back to notebooklm-py's
    stored credentials.

    Returns:
        AuthTokens ready for use with NotebookLMClient.

    Raises:
        RuntimeError: If no authentication is found.
    """
    # Try SmallClawLM's own auth
    if DEFAULT_STORAGE.exists():
        return await _load_auth(DEFAULT_STORAGE)

    # Fall back to notebooklm-py's stored auth
    try:
        storage = get_storage_path()
        storage_state = json.loads(storage.read_text())
        cookies = extract_cookies_from_storage(storage_state)
        csrf, session_id = await fetch_tokens(cookies)
        return AuthTokens(cookies=cookies, csrf_token=csrf, session_id=session_id)
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


def ensure_authenticated():
    """Check if the user is authenticated, raise if not."""
    try:
        asyncio.run(get_auth())
    except RuntimeError:
        raise
