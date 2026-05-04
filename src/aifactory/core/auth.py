"""
Auth utilities — JWT token creation/validation, password hashing, API key generation.

Used by:
  api/routes/auth.py  — register, login, refresh, logout
  api/routes/users.py — API key creation
  api/main.py         — get_current_user dependency

Token formats:
  Access token  — JWT signed with AIFACTORY_JWT_SECRET_KEY, 30-min TTL (default)
  Refresh token — opaque 48-byte URL-safe token; SHA-256 hash stored in DB
  API key       — "aif_<48 random chars>"; first 12 chars stored as prefix; SHA-256 hash in DB

All functions are synchronous and can be called from any context.
"""

from __future__ import annotations

import hashlib
import secrets
import string
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Password Hashing ──────────────────────────────────────────────────────────

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return bcrypt hash of plain-text password."""
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return _pwd_ctx.verify(plain, hashed)


# ── Settings helpers ─────────────────────────────────────────────────────────


def _secret() -> str:
    from aifactory.config.settings import get_settings

    return get_settings().jwt_secret_key


def _algo() -> str:
    from aifactory.config.settings import get_settings

    return get_settings().jwt_algorithm


# ── Access Token ─────────────────────────────────────────────────────────────


def create_access_token(
    user_id: str,
    email: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    Issue a signed JWT access token.

    Standard claims:
      sub   — user UUID
      email — user email
      exp   — expiry (configured via AIFACTORY_JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
      iat   — issued-at
      type  — "access"  (distinguishes from any other token type)
    """
    from aifactory.config.settings import get_settings

    expire = datetime.now(UTC) + timedelta(minutes=get_settings().jwt_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.now(UTC),
        "type": "access",
        **(extra or {}),
    }
    return jwt.encode(payload, _secret(), algorithm=_algo())


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and verify a JWT access token.

    Raises jose.JWTError on:
      - invalid signature
      - expired token
      - wrong token type
    """
    payload = jwt.decode(token, _secret(), algorithms=[_algo()])
    if payload.get("type") != "access":
        raise JWTError("not an access token")
    return payload


# ── Refresh Token ─────────────────────────────────────────────────────────────


def generate_refresh_token() -> tuple[str, str]:
    """
    Generate a new refresh token.

    Returns:
        (raw_token, token_hash)

    raw_token  — opaque 48-byte URL-safe string; returned to the client ONCE
    token_hash — SHA-256 hex digest; stored in the refresh_tokens table
    """
    raw = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, token_hash


def hash_token(raw: str) -> str:
    """Return SHA-256 hex digest of any token string."""
    return hashlib.sha256(raw.encode()).hexdigest()


# ── API Keys ──────────────────────────────────────────────────────────────────

_CHARSET = string.ascii_letters + string.digits


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        (raw_key, prefix, key_hash)

    raw_key  — "aif_<48 random chars>" — shown to the user exactly once
    prefix   — first 12 chars of raw_key (safe to display in key listings)
    key_hash — SHA-256 hex digest; the only value stored in the database
    """
    body = "".join(secrets.choice(_CHARSET) for _ in range(48))
    raw_key = f"aif_{body}"
    prefix = raw_key[:12]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hash used to look up an API key in the database."""
    return hashlib.sha256(raw_key.encode()).hexdigest()
