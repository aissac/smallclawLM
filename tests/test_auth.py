"""Tests for auth module."""
import pytest
from smallclawlm.auth import clear_cache


class TestAuthCache:
    """Test auth caching behavior."""

    def test_clear_cache_no_error(self):
        """clear_cache() should not raise even if no auth loaded."""
        clear_cache()
        # Second call should also work
        clear_cache()

    def test_auth_not_loaded_by_default(self):
        """Cache should start empty."""
        import smallclawlm.auth as auth_mod
        assert auth_mod._AUTH_CACHE is None
