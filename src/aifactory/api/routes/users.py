"""
User management routes — profile, API key CRUD.

GET    /api/users/me                 — alias for /api/auth/me (convenience)
PATCH  /api/users/me                 — update display name
GET    /api/users/api-keys           — list API keys
POST   /api/users/api-keys           — create a new API key (raw key returned once)
DELETE /api/users/api-keys/{key_id}  — revoke an API key
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from aifactory.core import database as db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["users"])


# ── Request / Response models ─────────────────────────────────────────────────


class UpdateProfileRequest(BaseModel):
    display_name: str | None = None


class CreateApiKeyRequest(BaseModel):
    name: str
    scopes: list[str] = []


class ApiKeyCreatedResponse(BaseModel):
    id: str
    name: str
    key: str          # raw key — shown ONCE, never retrievable again
    key_prefix: str
    scopes: list[str]
    created_at: str


class ApiKeyListItem(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool


# ── Dependency helper ─────────────────────────────────────────────────────────


async def _require_user(request: Request) -> dict:
    """Return the authenticated user dict or raise 401."""
    from aifactory.api.main import get_current_user

    ctx = await get_current_user(request)
    if ctx is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user = await db.get_user_by_id(ctx.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/api-keys", response_model=list[ApiKeyListItem])
async def list_keys(request: Request) -> list[ApiKeyListItem]:
    """List all API keys for the authenticated user."""
    user = await _require_user(request)
    rows = await db.list_api_keys(user["id"])
    return [
        ApiKeyListItem(
            id=str(r["id"]),
            name=r["name"],
            key_prefix=r["key_prefix"],
            scopes=list(r.get("scopes") or []),
            created_at=_iso(r.get("created_at")),
            last_used_at=_iso(r.get("last_used_at")),
            expires_at=_iso(r.get("expires_at")),
            is_active=r.get("is_active", True),
        )
        for r in rows
    ]


@router.post("/api-keys", response_model=ApiKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_key(body: CreateApiKeyRequest, request: Request) -> ApiKeyCreatedResponse:
    """
    Create a new API key.

    The full key is returned exactly once in this response — it is never
    retrievable again.  Store it securely (e.g. in your .env file).
    """
    from aifactory.core.auth import generate_api_key

    user = await _require_user(request)
    raw_key, prefix, key_hash = generate_api_key()

    row = await db.create_api_key_row(
        user_id=user["id"],
        name=body.name,
        key_prefix=prefix,
        key_hash=key_hash,
        scopes=body.scopes,
    )

    logger.info("API key created for user %s: %s", user["email"], prefix)
    return ApiKeyCreatedResponse(
        id=str(row["id"]),
        name=row["name"],
        key=raw_key,
        key_prefix=row["key_prefix"],
        scopes=list(row.get("scopes") or []),
        created_at=_iso(row.get("created_at")),
    )


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(key_id: str, request: Request) -> None:
    """Revoke (deactivate) an API key by its ID."""
    user = await _require_user(request)
    revoked = await db.revoke_api_key(key_id, user["id"])
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or does not belong to this account",
        )
    logger.info("API key %s revoked for user %s", key_id, user["email"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _iso(val: object) -> str | None:
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat().replace("+00:00", "Z")  # type: ignore[union-attr]
    return str(val)
