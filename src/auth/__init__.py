"""Beta auth — invited-user registry + FastAPI dependency.

This package mirrors :mod:`src.scopes` for the auth surface: a
YAML-loaded registry of users keyed by stable id, with optional
per-user scope whitelists. Wire details (cookie session, login route)
are in :mod:`src.web_api`.
"""

from .config import User, UserAuthError, UserRegistry, load_users

__all__ = [
    "User",
    "UserAuthError",
    "UserRegistry",
    "load_users",
]
