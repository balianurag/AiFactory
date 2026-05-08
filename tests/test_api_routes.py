"""Tests for API routes: auth, users, orgs — FastAPI TestClient with mocked DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aifactory.api.main import app
from aifactory.core import auth as _auth


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _user(
    user_id: str = "user-1",
    email: str = "alice@example.com",
    display_name: str | None = "Alice",
    is_active: bool = True,
) -> dict[str, Any]:
    return {
        "id": user_id,
        "email": email,
        "password_hash": "hashed-password-placeholder",
        "display_name": display_name,
        "created_at": "2024-01-01T00:00:00Z",
        "email_verified": False,
        "is_active": is_active,
    }


def _access_token(user_id: str = "user-1", email: str = "alice@example.com") -> str:
    return _auth.create_access_token(user_id, email)


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/auth/register
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegister:
    def test_register_success(self, client: TestClient) -> None:
        user = _user()
        with (
            patch("aifactory.api.routes.auth.db.get_user_by_email", AsyncMock(return_value=None)),
            patch("aifactory.api.routes.auth._auth.hash_password", return_value="hashed"),
            patch("aifactory.api.routes.auth.db.create_user", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.auth.db.create_refresh_token_row", AsyncMock(return_value=None)
            ),
        ):
            r = client.post(
                "/api/auth/register",
                json={"email": "alice@example.com", "password": "password123"},
            )
        assert r.status_code == 201
        body = r.json()
        assert "access_token" in body
        assert "refresh_token" in body
        assert body["token_type"] == "bearer"

    def test_register_duplicate_email(self, client: TestClient) -> None:
        with patch(
            "aifactory.api.routes.auth.db.get_user_by_email",
            AsyncMock(return_value=_user()),
        ):
            r = client.post(
                "/api/auth/register",
                json={"email": "alice@example.com", "password": "password123"},
            )
        assert r.status_code == 409

    def test_register_short_password(self, client: TestClient) -> None:
        r = client.post(
            "/api/auth/register",
            json={"email": "alice@example.com", "password": "abc"},
        )
        assert r.status_code == 422

    def test_register_invalid_email(self, client: TestClient) -> None:
        r = client.post(
            "/api/auth/register",
            json={"email": "not-an-email", "password": "password123"},
        )
        assert r.status_code == 422

    def test_register_with_display_name(self, client: TestClient) -> None:
        user = _user()
        with (
            patch("aifactory.api.routes.auth.db.get_user_by_email", AsyncMock(return_value=None)),
            patch("aifactory.api.routes.auth._auth.hash_password", return_value="hashed"),
            patch("aifactory.api.routes.auth.db.create_user", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.auth.db.create_refresh_token_row", AsyncMock(return_value=None)
            ),
        ):
            r = client.post(
                "/api/auth/register",
                json={
                    "email": "alice@example.com",
                    "password": "password123",
                    "display_name": "Alice",
                },
            )
        assert r.status_code == 201


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/auth/login
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogin:
    def test_login_success(self, client: TestClient) -> None:
        user = _user()
        with (
            patch(
                "aifactory.api.routes.auth.db.get_user_by_email", AsyncMock(return_value=user)
            ),
            patch("aifactory.api.routes.auth._auth.verify_password", return_value=True),
            patch("aifactory.api.routes.auth.db.touch_user_login", AsyncMock(return_value=None)),
            patch(
                "aifactory.api.routes.auth.db.create_refresh_token_row",
                AsyncMock(return_value=None),
            ),
        ):
            r = client.post(
                "/api/auth/login",
                json={"email": "alice@example.com", "password": "password123"},
            )
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body
        assert "refresh_token" in body

    def test_login_wrong_password(self, client: TestClient) -> None:
        user = _user()
        with (
            patch(
                "aifactory.api.routes.auth.db.get_user_by_email", AsyncMock(return_value=user)
            ),
            patch("aifactory.api.routes.auth._auth.verify_password", return_value=False),
        ):
            r = client.post(
                "/api/auth/login",
                json={"email": "alice@example.com", "password": "wrongpassword"},
            )
        assert r.status_code == 401

    def test_login_unknown_email(self, client: TestClient) -> None:
        with patch(
            "aifactory.api.routes.auth.db.get_user_by_email", AsyncMock(return_value=None)
        ):
            r = client.post(
                "/api/auth/login",
                json={"email": "nobody@example.com", "password": "password123"},
            )
        assert r.status_code == 401

    def test_login_disabled_account(self, client: TestClient) -> None:
        user = _user(is_active=False)
        with (
            patch(
                "aifactory.api.routes.auth.db.get_user_by_email", AsyncMock(return_value=user)
            ),
            patch("aifactory.api.routes.auth._auth.verify_password", return_value=True),
        ):
            r = client.post(
                "/api/auth/login",
                json={"email": "alice@example.com", "password": "password123"},
            )
        assert r.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/auth/refresh
# ═══════════════════════════════════════════════════════════════════════════════


class TestRefreshToken:
    def _make_token_row(self, revoked: bool = False, expired: bool = False) -> dict[str, Any]:
        # Use naive datetimes so the route's ternary (`replace(tzinfo=UTC)`) path fires correctly.
        expires_at = datetime.now() + timedelta(days=(-1 if expired else 30))
        return {
            "user_id": "user-1",
            "revoked_at": datetime.now() if revoked else None,
            "expires_at": expires_at,
        }

    def test_refresh_success(self, client: TestClient) -> None:
        raw, token_hash = _auth.generate_refresh_token()
        row = self._make_token_row()
        user = _user()
        with (
            patch(
                "aifactory.api.routes.auth.db.get_refresh_token_row", AsyncMock(return_value=row)
            ),
            patch("aifactory.api.routes.auth.db.revoke_refresh_token", AsyncMock(return_value=None)),
            patch(
                "aifactory.api.routes.auth.db.get_user_by_id", AsyncMock(return_value=user)
            ),
            patch(
                "aifactory.api.routes.auth.db.create_refresh_token_row",
                AsyncMock(return_value=None),
            ),
        ):
            r = client.post("/api/auth/refresh", json={"refresh_token": raw})
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_refresh_invalid_token(self, client: TestClient) -> None:
        with patch(
            "aifactory.api.routes.auth.db.get_refresh_token_row", AsyncMock(return_value=None)
        ):
            r = client.post("/api/auth/refresh", json={"refresh_token": "bad-token"})
        assert r.status_code == 401
        assert "Invalid" in r.json()["detail"]

    def test_refresh_revoked_token(self, client: TestClient) -> None:
        row = self._make_token_row(revoked=True)
        with patch(
            "aifactory.api.routes.auth.db.get_refresh_token_row", AsyncMock(return_value=row)
        ):
            r = client.post("/api/auth/refresh", json={"refresh_token": "some-token"})
        assert r.status_code == 401
        assert "revoked" in r.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/auth/logout
# ═══════════════════════════════════════════════════════════════════════════════


class TestLogout:
    def test_logout_success(self, client: TestClient) -> None:
        with patch(
            "aifactory.api.routes.auth.db.revoke_refresh_token", AsyncMock(return_value=None)
        ):
            r = client.post("/api/auth/logout", json={"refresh_token": "any-token"})
        assert r.status_code == 204

    def test_logout_missing_body(self, client: TestClient) -> None:
        r = client.post("/api/auth/logout", json={})
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/auth/me
# ═══════════════════════════════════════════════════════════════════════════════


class TestMe:
    def test_me_authenticated(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with patch("aifactory.api.routes.auth.db.get_user_by_id", AsyncMock(return_value=user)):
            r = client.get("/api/auth/me", headers=_auth_header(token))
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == "alice@example.com"
        assert body["id"] == "user-1"

    def test_me_no_auth(self, client: TestClient) -> None:
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_me_invalid_token(self, client: TestClient) -> None:
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer bad.token.here"})
        assert r.status_code == 401

    def test_me_user_not_found(self, client: TestClient) -> None:
        token = _access_token()
        with patch("aifactory.api.routes.auth.db.get_user_by_id", AsyncMock(return_value=None)):
            r = client.get("/api/auth/me", headers=_auth_header(token))
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/users/api-keys
# ═══════════════════════════════════════════════════════════════════════════════


def _api_key_row(key_id: str = "key-1") -> dict[str, Any]:
    return {
        "id": key_id,
        "name": "My Key",
        "key_prefix": "aif_abc123",
        "scopes": ["read"],
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        "last_used_at": None,
        "expires_at": None,
        "is_active": True,
    }


class TestListApiKeys:
    def test_list_keys_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        rows = [_api_key_row()]
        with (
            patch("aifactory.api.routes.users.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.users.db.list_api_keys", AsyncMock(return_value=rows)),
        ):
            r = client.get("/api/users/api-keys", headers=_auth_header(token))
        assert r.status_code == 200
        keys = r.json()
        assert len(keys) == 1
        assert keys[0]["name"] == "My Key"
        assert keys[0]["key_prefix"] == "aif_abc123"

    def test_list_keys_unauthenticated(self, client: TestClient) -> None:
        r = client.get("/api/users/api-keys")
        assert r.status_code == 401

    def test_list_keys_empty(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.users.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.users.db.list_api_keys", AsyncMock(return_value=[])),
        ):
            r = client.get("/api/users/api-keys", headers=_auth_header(token))
        assert r.status_code == 200
        assert r.json() == []


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/users/api-keys
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateApiKey:
    def test_create_key_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        row = {**_api_key_row(), "name": "CI Key", "scopes": ["read", "write"]}
        with (
            patch("aifactory.api.routes.users.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.users.db.create_api_key_row", AsyncMock(return_value=row)
            ),
        ):
            r = client.post(
                "/api/users/api-keys",
                json={"name": "CI Key", "scopes": ["read", "write"]},
                headers=_auth_header(token),
            )
        assert r.status_code == 201
        body = r.json()
        assert "key" in body  # raw key returned once
        assert body["name"] == "CI Key"
        assert body["key"].startswith("aif_")

    def test_create_key_unauthenticated(self, client: TestClient) -> None:
        r = client.post("/api/users/api-keys", json={"name": "X"})
        assert r.status_code == 401

    def test_create_key_no_name(self, client: TestClient) -> None:
        token = _access_token()
        r = client.post("/api/users/api-keys", json={}, headers=_auth_header(token))
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# DELETE /api/users/api-keys/{key_id}
# ═══════════════════════════════════════════════════════════════════════════════


class TestRevokeApiKey:
    def test_revoke_key_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.users.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.users.db.revoke_api_key", AsyncMock(return_value=True)),
        ):
            r = client.delete("/api/users/api-keys/key-1", headers=_auth_header(token))
        assert r.status_code == 204

    def test_revoke_key_not_found(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.users.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.users.db.revoke_api_key", AsyncMock(return_value=False)),
        ):
            r = client.delete("/api/users/api-keys/missing-key", headers=_auth_header(token))
        assert r.status_code == 404

    def test_revoke_key_unauthenticated(self, client: TestClient) -> None:
        r = client.delete("/api/users/api-keys/key-1")
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/orgs
# ═══════════════════════════════════════════════════════════════════════════════


def _org(org_id: str = "org-1", slug: str = "acme") -> dict[str, Any]:
    return {
        "id": org_id,
        "name": "Acme Corp",
        "slug": slug,
        "plan": "starter",
        "created_at": "2024-01-01T00:00:00Z",
        "role": "owner",
    }


class TestCreateOrg:
    def test_create_org_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        org = _org()
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.orgs.db.create_org", AsyncMock(return_value=org)),
        ):
            r = client.post(
                "/api/orgs",
                json={"name": "Acme Corp", "slug": "acme"},
                headers=_auth_header(token),
            )
        assert r.status_code == 201
        body = r.json()
        assert body["slug"] == "acme"
        assert body["role"] == "owner"

    def test_create_org_invalid_slug(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)):
            r = client.post(
                "/api/orgs",
                json={"name": "Bad", "slug": "UPPERCASE"},
                headers=_auth_header(token),
            )
        assert r.status_code == 422

    def test_create_org_duplicate_slug(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.create_org",
                AsyncMock(side_effect=Exception("unique constraint violated")),
            ),
        ):
            r = client.post(
                "/api/orgs",
                json={"name": "Acme", "slug": "acme"},
                headers=_auth_header(token),
            )
        assert r.status_code == 409

    def test_create_org_unauthenticated(self, client: TestClient) -> None:
        r = client.post("/api/orgs", json={"name": "X", "slug": "xorg"})
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/orgs
# ═══════════════════════════════════════════════════════════════════════════════


class TestListOrgs:
    def test_list_orgs_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        orgs = [
            {**_org(), "role": "owner"},
            {**_org("org-2", "beta-inc"), "name": "Beta Inc", "role": "member"},
        ]
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs)),
        ):
            r = client.get("/api/orgs", headers=_auth_header(token))
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_list_orgs_empty(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=[])),
        ):
            r = client.get("/api/orgs", headers=_auth_header(token))
        assert r.status_code == 200
        assert r.json() == []

    def test_list_orgs_unauthenticated(self, client: TestClient) -> None:
        r = client.get("/api/orgs")
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/orgs/{org_id}
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetOrg:
    def test_get_org_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        org = _org()
        orgs_list = [{**org, "role": "owner"}]
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs_list)
            ),
            patch("aifactory.api.routes.orgs.db.get_org", AsyncMock(return_value=org)),
        ):
            r = client.get("/api/orgs/org-1", headers=_auth_header(token))
        assert r.status_code == 200
        assert r.json()["slug"] == "acme"

    def test_get_org_not_member(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=[])),
        ):
            r = client.get("/api/orgs/org-1", headers=_auth_header(token))
        assert r.status_code == 403

    def test_get_org_unauthenticated(self, client: TestClient) -> None:
        r = client.get("/api/orgs/org-1")
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/orgs/{org_id}/members
# ═══════════════════════════════════════════════════════════════════════════════


class TestAddMember:
    def _admin_setup(self) -> tuple[dict, dict, list]:
        user = _user()
        org = _org()
        orgs_list = [{**org, "role": "owner"}]
        return user, org, orgs_list

    def test_add_member_success(self, client: TestClient) -> None:
        token = _access_token()
        user, org, orgs_list = self._admin_setup()
        target = _user("user-2", "bob@example.com", "Bob")
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs_list)
            ),
            patch("aifactory.api.routes.orgs.db.get_org", AsyncMock(return_value=org)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_by_email", AsyncMock(return_value=target)
            ),
            patch("aifactory.api.routes.orgs.db.add_org_member", AsyncMock(return_value=None)),
        ):
            r = client.post(
                "/api/orgs/org-1/members",
                json={"email": "bob@example.com", "role": "member"},
                headers=_auth_header(token),
            )
        assert r.status_code == 201
        assert "bob@example.com" in r.json()["message"]

    def test_add_member_user_not_found(self, client: TestClient) -> None:
        token = _access_token()
        user, org, orgs_list = self._admin_setup()
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs_list)
            ),
            patch("aifactory.api.routes.orgs.db.get_org", AsyncMock(return_value=org)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_by_email", AsyncMock(return_value=None)
            ),
        ):
            r = client.post(
                "/api/orgs/org-1/members",
                json={"email": "nobody@example.com"},
                headers=_auth_header(token),
            )
        assert r.status_code == 404

    def test_add_member_invalid_role(self, client: TestClient) -> None:
        token = _access_token()
        user, org, orgs_list = self._admin_setup()
        target = _user("user-2", "bob@example.com")
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs_list)
            ),
            patch("aifactory.api.routes.orgs.db.get_org", AsyncMock(return_value=org)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_by_email", AsyncMock(return_value=target)
            ),
        ):
            r = client.post(
                "/api/orgs/org-1/members",
                json={"email": "bob@example.com", "role": "superuser"},
                headers=_auth_header(token),
            )
        assert r.status_code == 422

    def test_add_member_non_admin(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        org = _org()
        orgs_list = [{**org, "role": "member"}]  # not admin
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs_list)
            ),
            patch("aifactory.api.routes.orgs.db.get_org", AsyncMock(return_value=org)),
        ):
            r = client.post(
                "/api/orgs/org-1/members",
                json={"email": "bob@example.com"},
                headers=_auth_header(token),
            )
        assert r.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/orgs/{org_id}/members
# ═══════════════════════════════════════════════════════════════════════════════


class TestListMembers:
    def test_list_members_success(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        org = _org()
        orgs_list = [{**org, "role": "owner"}]
        members = [
            {
                "id": "user-1",
                "email": "alice@example.com",
                "display_name": "Alice",
                "role": "owner",
                "joined_at": datetime(2024, 1, 1, tzinfo=UTC),
            }
        ]
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch(
                "aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=orgs_list)
            ),
            patch("aifactory.api.routes.orgs.db.get_org", AsyncMock(return_value=org)),
            patch(
                "aifactory.api.routes.orgs.db.list_org_members", AsyncMock(return_value=members)
            ),
        ):
            r = client.get("/api/orgs/org-1/members", headers=_auth_header(token))
        assert r.status_code == 200
        result = r.json()
        assert len(result) == 1
        assert result[0]["email"] == "alice@example.com"
        assert result[0]["role"] == "owner"

    def test_list_members_not_member(self, client: TestClient) -> None:
        token = _access_token()
        user = _user()
        with (
            patch("aifactory.api.routes.orgs.db.get_user_by_id", AsyncMock(return_value=user)),
            patch("aifactory.api.routes.orgs.db.get_user_orgs", AsyncMock(return_value=[])),
        ):
            r = client.get("/api/orgs/org-1/members", headers=_auth_header(token))
        assert r.status_code == 403

    def test_list_members_unauthenticated(self, client: TestClient) -> None:
        r = client.get("/api/orgs/org-1/members")
        assert r.status_code == 401
