"""User auth config — invited users + token sha256s + scope whitelist.

Mirrors :mod:`src.scopes.config`: same loader-precedence pattern,
same strict validation. The token plaintext lives off-system with each
user; this module only stores SHA-256s for lookup.

The loader honours the ``DEKA_USERS_FILE`` env var so deployments can
point at an alternate path without editing the repo file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.scopes import ScopeRegistry, load_scopes

_ENV_PATH = "DEKA_USERS_FILE"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRIMARY = _REPO_ROOT / "users.yaml"
_FALLBACK = _REPO_ROOT / "users.yaml.example"

_REQUIRED_KEYS = frozenset({"id", "token_sha256"})
_OPTIONAL_KEYS = frozenset({"allowed_scopes"})
_KNOWN_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UserAuthError(RuntimeError):
    """Raised when the users file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class User:
    """A single invited user.

    ``id`` is both the display label and the stable wire key. Renaming
    an ``id`` orphans that user's on-disk sessions (which live under
    ``runs/<id>/``) and is therefore treated like renaming a
    :class:`Scope.name`: a breaking change.

    ``token_sha256`` is the lowercase hex SHA-256 of the user's bearer
    token. The plaintext is never stored.

    ``allowed_scopes`` whitelists the scope names this user may
    pick from at session start. ``None`` means "every scope declared
    in scopes.yaml".
    """

    id: str
    token_sha256: str
    allowed_scopes: tuple[str, ...] | None


@dataclass(frozen=True)
class UserRegistry:
    """Ordered collection of invited users.

    Order matches the YAML source. Lookup is by id (display key) or by
    token sha (auth key); both are O(n) over the small beta audience —
    no need for a hash index.
    """

    users: tuple[User, ...]

    def ids(self) -> list[str]:
        return [u.id for u in self.users]

    def get(self, user_id: str) -> User:
        for user in self.users:
            if user.id == user_id:
                return user
        raise UserAuthError(f"Unknown user id {user_id!r}; available: {self.ids()}")

    def find_by_token_sha(self, sha: str) -> User | None:
        """Return the user whose ``token_sha256`` matches, or ``None``.

        Comparison is constant-time-per-entry but the iteration order
        is plaintext; for beta audiences (a handful of users) that is
        adequate. If the audience grows past ~100, swap for a dict
        keyed by sha.
        """
        for user in self.users:
            if user.token_sha256 == sha:
                return user
        return None

    def __iter__(self):
        return iter(self.users)

    def __len__(self) -> int:
        return len(self.users)


def _resolve_path(explicit: Path | None) -> Path:
    """Pick which file to read.

    Precedence: explicit argument > ``DEKA_USERS_FILE`` env >
    repo-local ``users.yaml`` > committed ``users.yaml.example``.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get(_ENV_PATH)
    if env:
        return Path(env)
    if _PRIMARY.exists():
        return _PRIMARY
    return _FALLBACK


def load_users(
    path: Path | None = None,
    *,
    scope_registry: ScopeRegistry | None = None,
) -> UserRegistry:
    """Load and validate the users config.

    Cross-checks every ``allowed_scopes`` entry against the available
    scope names from :mod:`src.scopes`. Pass ``scope_registry``
    explicitly to validate against a non-default scopes file (tests
    do this); otherwise the default ``scopes.yaml`` is loaded.

    Raises :class:`UserAuthError` on any missing-file, parse, or
    schema failure.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise UserAuthError(f"users file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise UserAuthError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise UserAuthError(f"{resolved}: top-level mapping with key 'users' required")
    if "users" not in raw:
        raise UserAuthError(f"{resolved}: missing top-level 'users' key")
    entries = raw["users"]
    if not isinstance(entries, list) or not entries:
        raise UserAuthError(f"{resolved}: 'users' must be a non-empty list")

    if scope_registry is None:
        scope_registry = load_scopes()
    declared_scopes = set(scope_registry.names())

    users: list[User] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise UserAuthError(f"{resolved}: users[{index}] must be a mapping")
        missing = _REQUIRED_KEYS - entry.keys()
        if missing:
            raise UserAuthError(
                f"{resolved}: users[{index}] missing required keys: {sorted(missing)}"
            )
        unknown = entry.keys() - _KNOWN_KEYS
        if unknown:
            raise UserAuthError(
                f"{resolved}: users[{index}] has unknown keys: {sorted(unknown)}"
            )

        for key in ("id", "token_sha256"):
            value = entry[key]
            if not isinstance(value, str) or not value.strip():
                raise UserAuthError(
                    f"{resolved}: users[{index}].{key} must be a non-empty string"
                )

        user_id = entry["id"]
        token_sha = entry["token_sha256"]
        # Strict format: enforce lowercase hex rather than silently
        # normalising. ``hashlib.sha256(...).hexdigest()`` always
        # returns lowercase, so an uppercase value in the YAML is
        # almost certainly a copy-paste mistake from an external
        # tool — we'd rather the operator notice now than at login.
        if not _SHA256_RE.match(token_sha):
            raise UserAuthError(
                f"{resolved}: users[{index}].token_sha256 must be a "
                f"64-char lowercase hex SHA-256 digest"
            )

        if user_id in seen:
            raise UserAuthError(f"{resolved}: duplicate user id {user_id!r}")
        seen.add(user_id)

        allowed: tuple[str, ...] | None
        raw_scopes = entry.get("allowed_scopes")
        if raw_scopes is None:
            allowed = None
        else:
            if not isinstance(raw_scopes, list) or not raw_scopes:
                raise UserAuthError(
                    f"{resolved}: users[{index}].allowed_scopes must be "
                    f"a non-empty list, or omitted/null to allow every "
                    f"scope"
                )
            seen_scopes: set[str] = set()
            for scope_index, scope_name in enumerate(raw_scopes):
                if not isinstance(scope_name, str) or not scope_name.strip():
                    raise UserAuthError(
                        f"{resolved}: users[{index}]."
                        f"allowed_scopes[{scope_index}] must be a "
                        f"non-empty string"
                    )
                if scope_name in seen_scopes:
                    raise UserAuthError(
                        f"{resolved}: users[{index}]."
                        f"allowed_scopes contains duplicate "
                        f"{scope_name!r}"
                    )
                if scope_name not in declared_scopes:
                    raise UserAuthError(
                        f"{resolved}: users[{index}]."
                        f"allowed_scopes[{scope_index}] = "
                        f"{scope_name!r} is not declared in "
                        f"scopes.yaml; available: "
                        f"{sorted(declared_scopes)}"
                    )
                seen_scopes.add(scope_name)
            allowed = tuple(raw_scopes)

        users.append(User(id=user_id, token_sha256=token_sha, allowed_scopes=allowed))

    return UserRegistry(users=tuple(users))


def _resolved_path_for_test(explicit: Path | None = None) -> Path:
    """Internal accessor used by tests to assert path-resolution priority."""
    return _resolve_path(explicit)


__all__ = [
    "User",
    "UserAuthError",
    "UserRegistry",
    "load_users",
]
