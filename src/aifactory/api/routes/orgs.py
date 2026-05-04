"""
Organization routes — create org, manage members.

POST /api/orgs                        — create an organization
GET  /api/orgs/{org_id}               — get org details (member only)
GET  /api/orgs                        — list orgs the user belongs to
POST /api/orgs/{org_id}/members       — add a member by email
GET  /api/orgs/{org_id}/members       — list members
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from aifactory.core import database as db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orgs", tags=["organizations"])

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{1,48}[a-z0-9]$")


# ── Request / Response models ─────────────────────────────────────────────────


class CreateOrgRequest(BaseModel):
    name: str
    slug: str
    plan: str = "starter"


class AddMemberRequest(BaseModel):
    email: str
    role: str = "member"


class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    plan: str
    created_at: str
    role: str | None = None  # caller's membership role (list endpoint only)


class MemberResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    joined_at: str


# ── Dependency helper ─────────────────────────────────────────────────────────


async def _require_user(request: Request) -> dict:
    from aifactory.api.main import get_current_user

    ctx = await get_current_user(request)
    if ctx is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = await db.get_user_by_id(ctx.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


async def _require_org_member(request: Request, org_id: str) -> tuple[dict, dict]:
    """Return (user, org) — raises 403 if user is not a member."""
    user = await _require_user(request)
    orgs = await db.get_user_orgs(user["id"])
    org_map = {str(o["id"]): o for o in orgs}
    if org_id not in org_map:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this organisation"
        )
    org = await db.get_org(org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisation not found")
    return user, org


async def _require_org_admin(request: Request, org_id: str) -> tuple[dict, dict]:
    """Return (user, org) — raises 403 if user is not owner/admin."""
    user = await _require_user(request)
    orgs = await db.get_user_orgs(user["id"])
    org_map = {str(o["id"]): o for o in orgs}
    if org_id not in org_map or org_map[org_id]["role"] not in ("owner", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    org = await db.get_org(org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisation not found")
    return user, org


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=OrgResponse, status_code=status.HTTP_201_CREATED)
async def create_org(body: CreateOrgRequest, request: Request) -> OrgResponse:
    """Create a new organisation.  The caller becomes its owner."""
    user = await _require_user(request)

    if not _SLUG_RE.match(body.slug):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Slug must be 3–50 lowercase alphanumeric chars or hyphens (no leading/trailing hyphens)",
        )

    try:
        org = await db.create_org(
            name=body.name,
            slug=body.slug,
            owner_id=user["id"],
            plan=body.plan,
        )
    except Exception as exc:
        if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Organisation slug '{body.slug}' is already taken",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    logger.info("Organisation created: %s (owner=%s)", body.slug, user["email"])
    return OrgResponse(
        id=str(org["id"]),
        name=org["name"],
        slug=org["slug"],
        plan=org["plan"],
        created_at=_iso(org.get("created_at")),
        role="owner",
    )


@router.get("", response_model=list[OrgResponse])
async def list_orgs(request: Request) -> list[OrgResponse]:
    """List all organisations the authenticated user belongs to."""
    user = await _require_user(request)
    orgs = await db.get_user_orgs(user["id"])
    return [
        OrgResponse(
            id=str(o["id"]),
            name=o["name"],
            slug=o["slug"],
            plan=o["plan"],
            created_at=_iso(o.get("created_at")),
            role=o.get("role"),
        )
        for o in orgs
    ]


@router.get("/{org_id}", response_model=OrgResponse)
async def get_org(org_id: str, request: Request) -> OrgResponse:
    """Return organisation details. Requires membership."""
    user, org = await _require_org_member(request, org_id)
    orgs = await db.get_user_orgs(user["id"])
    role = next((o["role"] for o in orgs if str(o["id"]) == org_id), None)
    return OrgResponse(
        id=str(org["id"]),
        name=org["name"],
        slug=org["slug"],
        plan=org["plan"],
        created_at=_iso(org.get("created_at")),
        role=role,
    )


@router.post("/{org_id}/members", status_code=status.HTTP_201_CREATED)
async def add_member(org_id: str, body: AddMemberRequest, request: Request) -> dict:
    """Add a user to the organisation by email. Requires admin or owner role."""
    _user, _org = await _require_org_admin(request, org_id)

    target = await db.get_user_by_email(body.email)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account found with email '{body.email}'",
        )

    valid_roles = ("owner", "admin", "member")
    if body.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Role must be one of: {', '.join(valid_roles)}",
        )

    await db.add_org_member(target["id"], org_id, body.role)
    logger.info("User %s added to org %s as %s", body.email, org_id, body.role)
    return {"message": f"Added {body.email} as {body.role}"}


@router.get("/{org_id}/members", response_model=list[MemberResponse])
async def list_members(org_id: str, request: Request) -> list[MemberResponse]:
    """List all members of an organisation. Requires membership."""
    await _require_org_member(request, org_id)
    members = await db.list_org_members(org_id)
    return [
        MemberResponse(
            id=str(m["id"]),
            email=m["email"],
            display_name=m.get("display_name"),
            role=m["role"],
            joined_at=_iso(m.get("joined_at")) or "",
        )
        for m in members
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _iso(val: object) -> str:
    if val is None:
        return ""
    if hasattr(val, "isoformat"):
        return val.isoformat().replace("+00:00", "Z")  # type: ignore[union-attr]
    return str(val)
