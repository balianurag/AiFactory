"""
Auth routes — register, login, refresh, logout, me.

POST /api/auth/register  — create account
POST /api/auth/login     — get access + refresh token
POST /api/auth/refresh   — exchange refresh token for new access token
POST /api/auth/logout    — revoke the supplied refresh token
GET  /api/auth/me        — return the current user profile
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, field_validator

from aifactory.config.settings import get_settings
from aifactory.core import auth as _auth
from aifactory.core import database as db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Request / Response models ─────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = None

    @field_validator("password")
    @classmethod
    def _password_length(cls, v: str) -> str:
        min_len = get_settings().password_min_length
        if len(v) < min_len:
            raise ValueError(f"Password must be at least {min_len} characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access token lifetime in seconds


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    created_at: str
    email_verified: bool


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "")
    )


def _ua(request: Request) -> str:
    return request.headers.get("User-Agent", "")[:200]


async def _issue_tokens(user_id: str, email: str, request: Request) -> TokenResponse:
    """Create access + refresh tokens, persist refresh token to DB."""
    settings = get_settings()

    access_token = _auth.create_access_token(user_id, email)
    raw_refresh, token_hash = _auth.generate_refresh_token()

    expires_at = datetime.now(UTC) + timedelta(days=settings.jwt_refresh_token_expire_days)
    await db.create_refresh_token_row(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        ip_addr=_ip(request),
        user_agent=_ua(request),
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, request: Request) -> TokenResponse:
    """Create a new user account and return access + refresh tokens."""
    existing = await db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    password_hash = _auth.hash_password(body.password)
    try:
        user = await db.create_user(
            email=body.email,
            password_hash=password_hash,
            display_name=body.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    logger.info("New user registered: %s", body.email)
    return await _issue_tokens(user["id"], user["email"], request)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request) -> TokenResponse:
    """Authenticate with email + password, return access + refresh tokens."""
    user = await db.get_user_by_email(body.email)
    if not user or not _auth.verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled — contact support",
        )

    await db.touch_user_login(user["id"])
    return await _issue_tokens(user["id"], user["email"], request)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, request: Request) -> TokenResponse:
    """Exchange a valid refresh token for a new access + refresh token pair."""
    token_hash = _auth.hash_token(body.refresh_token)
    row = await db.get_refresh_token_row(token_hash)

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )

    if row.get("revoked_at") is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked"
        )

    expires_at = row.get("expires_at")
    if (
        expires_at and datetime.now(UTC) > expires_at.replace(tzinfo=UTC)
        if not expires_at.tzinfo
        else expires_at
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired"
        )

    # Rotate: revoke old token
    await db.revoke_refresh_token(token_hash)

    user_id = str(row["user_id"])
    user = await db.get_user_by_id(user_id)
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or disabled"
        )

    return await _issue_tokens(user_id, user["email"], request)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest) -> None:
    """Revoke the supplied refresh token (client should also discard the access token)."""
    token_hash = _auth.hash_token(body.refresh_token)
    await db.revoke_refresh_token(token_hash)


@router.get("/me", response_model=UserResponse)
async def me(request: Request) -> UserResponse:
    """Return the currently authenticated user's profile."""
    from aifactory.api.main import get_current_user

    user_ctx = await get_current_user(request)
    if user_ctx is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user = await db.get_user_by_id(user_ctx.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(
        id=user["id"],
        email=user["email"],
        display_name=user.get("display_name"),
        created_at=user.get("created_at", ""),
        email_verified=user.get("email_verified", False),
    )
