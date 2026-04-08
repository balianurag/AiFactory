#!/usr/bin/env python3
"""
Automated Integration Connector — Connects all configured toolkits in one shot.

For each toolkit in composio.yaml that isn't already connected, this script:
  1. Creates (or reuses) a Composio auth config
  2. Initiates the OAuth connection
  3. Prints the authorization URL
  4. Optionally opens all URLs in the browser
  5. Polls until connections are active (or timeout)

═══════════════════════════════════════════════════════════════════════════════
USAGE:
    # Connect all toolkits from composio.yaml (interactive — opens browser):
    uv run python scripts/connect_integrations.py

    # Connect specific toolkits only:
    uv run python scripts/connect_integrations.py notion gmail linear

    # Just print URLs (don't open browser):
    uv run python scripts/connect_integrations.py --no-open

    # Check status of all connections:
    uv run python scripts/connect_integrations.py --status

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

POLL_INTERVAL = 3  # seconds
POLL_TIMEOUT = 120  # total seconds to wait per batch


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}{RESET}\n")


def get_client():
    from composio import Composio

    key = os.environ.get("COMPOSIO_API_KEY", "")
    if not key:
        print(f"{RED}✗ COMPOSIO_API_KEY not set. Add it to .env{RESET}")
        sys.exit(1)
    return Composio(api_key=key)


def get_configured_toolkits() -> list[str]:
    """Load enabled toolkits from composio.yaml."""
    from aifactory.integrations.registry import IntegrationRegistry

    configs = IntegrationRegistry.load_configs()
    for cfg in configs:
        if cfg.name == "composio":
            return cfg.enabled_toolkits
    return ["github", "gitlab", "slack", "jira", "notion", "gmail"]


def get_connected_toolkits(client) -> dict[str, str]:
    """Return {toolkit_slug: status} for all connected accounts."""
    resp = client.connected_accounts.list()
    connected: dict[str, str] = {}
    for a in resp.items:
        d = a.model_dump() if hasattr(a, "model_dump") else {}
        tk = d.get("toolkit", {}).get("slug", "")
        status = d.get("status", "UNKNOWN")
        if tk:
            # Keep the best status (ACTIVE > INITIATED > other)
            existing = connected.get(tk, "")
            if status == "ACTIVE" or existing != "ACTIVE":
                connected[tk] = status
    return connected


def get_auth_configs(client) -> dict[str, str]:
    """Return {toolkit_slug: auth_config_id} for existing configs."""
    resp = client.auth_configs.list()
    configs: dict[str, str] = {}
    for c in resp.items:
        d = c.model_dump() if hasattr(c, "model_dump") else {}
        tk = d.get("toolkit", {}).get("slug", "")
        cid = d.get("id", "")
        if tk and cid:
            configs[tk] = cid
    return configs


def create_auth_config(client, toolkit: str) -> str:
    """Create a Composio-managed auth config for a toolkit. Returns auth_config_id."""
    try:
        cfg = client.auth_configs.create(
            toolkit=toolkit,
            options={"type": "use_composio_managed_auth", "name": f"{toolkit}-aifactory"},
        )
        return cfg.id
    except Exception as e:
        # May already exist or toolkit not supported
        err = str(e)
        if "already exists" in err.lower():
            # Fetch existing
            configs = get_auth_configs(client)
            return configs.get(toolkit, "")
        raise


def initiate_connection(client, toolkit: str, auth_config_id: str) -> tuple[str, str]:
    """Initiate an OAuth connection. Returns (connection_id, redirect_url)."""
    conn = client.connected_accounts.initiate(
        user_id="default",
        auth_config_id=auth_config_id,
    )
    return conn.id, conn.redirect_url


def show_status(client, toolkits: list[str]) -> None:
    """Print the current status of all configured toolkits."""
    header("Integration Connection Status")
    connected = get_connected_toolkits(client)

    for tk in toolkits:
        status = connected.get(tk, "NOT CONNECTED")
        if status == "ACTIVE":
            icon = f"{GREEN}✓{RESET}"
        elif status == "INITIATED":
            icon = f"{YELLOW}⟳{RESET}"
        else:
            icon = f"{RED}✗{RESET}"
        print(f"  {icon} {tk:20s} {status}")

    active = sum(1 for s in connected.values() if s == "ACTIVE")
    total = len(toolkits)
    print(f"\n  {DIM}{active}/{total} toolkits connected{RESET}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    no_open = "--no-open" in sys.argv
    status_only = "--status" in sys.argv

    client = get_client()
    all_toolkits = get_configured_toolkits()
    target_toolkits = args if args else all_toolkits

    if status_only:
        show_status(client, all_toolkits)
        return

    header("AI Factory — Automated Integration Connector")
    print(f"  {DIM}Configured toolkits: {', '.join(all_toolkits)}{RESET}")
    print(f"  {DIM}Target: {', '.join(target_toolkits)}{RESET}\n")

    # ── Check what's already connected ────────────────────────────────
    connected = get_connected_toolkits(client)
    auth_configs = get_auth_configs(client)

    already_active = []
    need_connection = []

    for tk in target_toolkits:
        status = connected.get(tk, "")
        if status == "ACTIVE":
            already_active.append(tk)
        else:
            need_connection.append(tk)

    if already_active:
        print(f"  {GREEN}Already connected:{RESET} {', '.join(already_active)}")
    if not need_connection:
        print(f"\n  {GREEN}All target toolkits are already connected!{RESET}")
        show_status(client, all_toolkits)
        return

    print(f"  {YELLOW}Need connection:{RESET} {', '.join(need_connection)}\n")

    # ── Create auth configs + initiate connections ────────────────────
    pending: list[tuple[str, str, str]] = []  # (toolkit, conn_id, url)

    for tk in need_connection:
        try:
            # Get or create auth config
            if tk in auth_configs:
                config_id = auth_configs[tk]
                print(f"  {DIM}Reusing auth config for {tk}: {config_id}{RESET}")
            else:
                config_id = create_auth_config(client, tk)
                print(f"  {GREEN}✓{RESET} Created auth config for {tk}: {config_id}")

            if not config_id:
                print(f"  {RED}✗ Failed to get auth config for {tk} — skipping{RESET}")
                continue

            # Initiate OAuth connection
            conn_id, url = initiate_connection(client, tk, config_id)
            pending.append((tk, conn_id, url))
            print(f"  {GREEN}✓{RESET} {tk}: OAuth URL ready")

        except Exception as e:
            print(f"  {RED}✗ {tk}: {e}{RESET}")

    if not pending:
        print(f"\n  {RED}No connections could be initiated.{RESET}")
        return

    # ── Print all URLs ────────────────────────────────────────────────
    print(f"\n{BOLD}{'─' * 64}{RESET}")
    print(f"{BOLD}  Authorization URLs (complete each in your browser):{RESET}")
    print(f"{'─' * 64}\n")

    for tk, _conn_id, url in pending:
        print(f"  {BOLD}{tk}{RESET}:")
        print(f"    {CYAN}{url}{RESET}\n")

    # ── Open in browser ───────────────────────────────────────────────
    if not no_open:
        print(f"{DIM}  Opening URLs in browser...{RESET}\n")
        for _tk, _, url in pending:
            try:
                subprocess.run(["open", url], check=True, capture_output=True)
                time.sleep(0.5)  # Stagger to avoid tab flood
            except Exception:
                pass  # Non-macOS or no browser

    # ── Poll for completion ───────────────────────────────────────────
    print(f"{BOLD}  Waiting for authorizations (timeout: {POLL_TIMEOUT}s)...{RESET}\n")
    pending_set = {tk for tk, _, _ in pending}
    completed: set[str] = set()
    start = time.time()

    while pending_set - completed and (time.time() - start) < POLL_TIMEOUT:
        time.sleep(POLL_INTERVAL)
        current = get_connected_toolkits(client)
        for tk in list(pending_set - completed):
            if current.get(tk) == "ACTIVE":
                completed.add(tk)
                elapsed = time.time() - start
                print(f"  {GREEN}✓ {tk} connected!{RESET} ({elapsed:.0f}s)")

    still_pending = pending_set - completed
    if still_pending:
        print(f"\n  {YELLOW}⏱ Timed out waiting for: {', '.join(still_pending)}{RESET}")
        print(f"  {DIM}Complete the OAuth flow in your browser, then run:{RESET}")
        print(f"  {DIM}  uv run python scripts/connect_integrations.py --status{RESET}")

    # ── Final status ──────────────────────────────────────────────────
    show_status(client, all_toolkits)


if __name__ == "__main__":
    main()
