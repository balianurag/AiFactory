#!/usr/bin/env python3
"""Live integration test — validates the full Composio ↔ IntegrationRegistry stack
against real APIs (GitHub, Slack, Jira, etc.).

═══════════════════════════════════════════════════════════════════════════════
SETUP (one-time):
    1. Get your Composio API key from https://app.composio.dev
       echo 'COMPOSIO_API_KEY=your_key_here' >> .env

    2. Install the optional integrations dependency:
       uv sync --extra integrations

    3. Connect the apps you want to test (interactive OAuth flow):
       uv run composio add github
       uv run composio add slack
       uv run composio add jira          # optional
       uv run composio add notion        # optional

    4. Verify connected apps:
       uv run composio apps

═══════════════════════════════════════════════════════════════════════════════
USAGE:
    # Run all tests (skips apps you haven't connected):
    uv run python scripts/test_integrations_live.py

    # Run specific toolkit tests only:
    uv run python scripts/test_integrations_live.py github
    uv run python scripts/test_integrations_live.py github slack

    # Dry-run (connect + list tools, but don't execute anything):
    uv run python scripts/test_integrations_live.py --dry-run

═══════════════════════════════════════════════════════════════════════════════
WHAT IT TESTS:
    1. ComposioProvider.connect() — API key validation, SDK init
    2. ComposioProvider.health_check() — can we talk to Composio?
    3. fetch_available_tools() — enumerate real tools per toolkit
    4. execute_action() — run a SAFE read-only action on each connected app
    5. IntegrationRegistry full flow — register → connect → execute → results
    6. extract_action_blocks() + execute_integration_actions() — E2E text protocol
    7. API endpoint smoke test (if server running) — /api/integrations
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aifactory.integrations.providers.composio import ComposioProvider

# Load .env before any project imports
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── Pretty output ──────────────────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}{RESET}\n")


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    global failed
    failed += 1
    print(f"  {RED}✗{RESET} {msg}")


def skip(msg: str) -> None:
    global skipped
    skipped += 1
    print(f"  {YELLOW}⊘{RESET} {DIM}{msg}{RESET}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


# ── Safe read-only actions per toolkit (no side effects) ───────────────
SAFE_ACTIONS: dict[str, tuple[str, dict]] = {
    "github": ("GITHUB_AUTH_USER_DOCKER_CONFLICT_PACKAGES_LIST", {}),
    "gitlab": ("GITLAB_GET_PROJECTS", {}),
    "slack": ("SLACK_LIST_CHANNELS", {"limit": 1}),
    "jira": ("JIRA_GET_PROJECTS", {}),
    "notion": ("NOTION_LIST_DATABASES", {}),
    "gmail": ("GMAIL_LIST_LABELS", {}),
    "linear": ("LINEAR_LIST_TEAMS", {}),
    "confluence": ("CONFLUENCE_LIST_SPACES", {"limit": 1}),
    "googledrive": ("GOOGLEDRIVE_LIST_FILES", {"page_size": 1}),
    "googlecalendar": ("GOOGLECALENDAR_LIST_CALENDARS", {}),
}


async def test_provider_lifecycle(dry_run: bool = False) -> ComposioProvider | None:
    """Test 1: ComposioProvider connect + health check."""
    from aifactory.integrations.providers.composio import ComposioProvider

    header("Test 1: ComposioProvider Lifecycle")

    api_key = os.getenv("COMPOSIO_API_KEY", "")
    if not api_key:
        fail("COMPOSIO_API_KEY not set in .env — see setup instructions above")
        return None
    ok(f"COMPOSIO_API_KEY found ({api_key[:8]}...)")

    provider = ComposioProvider()
    info(
        f"Config: {len(provider.config.enabled_toolkits)} enabled toolkits: "
        f"{', '.join(provider.config.enabled_toolkits)}"
    )

    t0 = time.perf_counter()
    connected = await provider.connect()
    dt = time.perf_counter() - t0

    if connected:
        ok(f"Connected in {dt:.2f}s (status: {provider.status})")
    else:
        fail(f"Connection failed in {dt:.2f}s (status: {provider.status})")
        return None

    t0 = time.perf_counter()
    healthy = await provider.health_check()
    dt = time.perf_counter() - t0

    if healthy:
        ok(f"Health check passed in {dt:.2f}s")
    else:
        fail(f"Health check failed in {dt:.2f}s")

    return provider


async def test_tool_discovery(
    provider: ComposioProvider,
    toolkits_filter: list[str] | None = None,
) -> dict[str, list[str]]:
    """Test 2: Fetch available tools from Composio."""
    header("Test 2: Tool Discovery")

    target = toolkits_filter or provider.config.enabled_toolkits
    info(f"Fetching tools for: {', '.join(target)}")

    t0 = time.perf_counter()
    tools = await provider.fetch_available_tools(target)
    dt = time.perf_counter() - t0

    total_tools = sum(len(v) for v in tools.values())
    ok(f"Discovered {total_tools} tools across {len(tools)} toolkits in {dt:.2f}s")

    for tk, slugs in sorted(tools.items()):
        if slugs:
            preview = ", ".join(slugs[:5])
            extra = f" (+{len(slugs) - 5} more)" if len(slugs) > 5 else ""
            info(f"  {tk}: {len(slugs)} tools — {preview}{extra}")
        else:
            skip(f"  {tk}: no tools found (app may not be connected — run `composio add {tk}`)")

    # Test prompt summary generation
    summary = provider.get_tools_summary_for_prompt()
    if summary:
        ok(f"Prompt summary generated ({len(summary)} chars)")
    else:
        fail("Prompt summary is empty")

    return tools


async def test_safe_actions(
    provider: ComposioProvider,
    tools: dict[str, list[str]],
    toolkits_filter: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Test 3: Execute safe read-only actions."""
    header("Test 3: Safe Action Execution" + (" (DRY RUN)" if dry_run else ""))

    target = toolkits_filter or list(tools.keys())

    for tk in sorted(target):
        slugs = tools.get(tk, [])
        if not slugs:
            skip(f"{tk}: no tools available — skipping")
            continue

        action, params = SAFE_ACTIONS.get(tk, (None, None))
        if action is None:
            skip(f"{tk}: no safe test action defined — skipping")
            continue

        # Verify the action slug actually exists
        if action not in slugs:
            # Try to find a similar read-only action
            read_actions = [s for s in slugs if "LIST" in s or "GET" in s]
            if read_actions:
                action = read_actions[0]
                params = {}
                info(f"{tk}: using alternative action: {action}")
            else:
                skip(f"{tk}: safe action {action} not found in available tools")
                continue

        if dry_run:
            skip(f"{tk}: would execute {action}({json.dumps(params)}) — dry run")
            continue

        t0 = time.perf_counter()
        result = await provider.execute_action(action, params)
        dt = time.perf_counter() - t0

        if result.success:
            ok(f"{tk}/{action}: success in {dt:.2f}s")
            if result.url:
                info(f"  URL: {result.url}")
            if result.data:
                data_preview = json.dumps(result.data, default=str)[:200]
                info(f"  Data: {data_preview}")
        else:
            fail(f"{tk}/{action}: {result.error} ({dt:.2f}s)")


async def test_registry_flow(
    toolkits_filter: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Test 4: Full IntegrationRegistry → ComposioProvider flow."""
    header("Test 4: IntegrationRegistry E2E")

    from aifactory.integrations.providers.composio import ComposioProvider
    from aifactory.integrations.registry import IntegrationRegistry

    # Reset to avoid singleton pollution from earlier tests
    IntegrationRegistry.reset()
    registry = IntegrationRegistry()

    provider = ComposioProvider()
    registry.register(provider)
    ok("Registered ComposioProvider in registry")

    results = await registry.connect_all()
    if results.get("composio"):
        ok("Registry.connect_all() → composio connected")
    else:
        fail("Registry.connect_all() → composio failed")
        return

    # Summary
    summary = registry.summary()
    info(f"Registry summary: {json.dumps(summary, default=str)}")

    # Execute via registry (mirrors what the pipeline does)
    tk = (toolkits_filter or ["github"])[0]
    action, params = SAFE_ACTIONS.get(tk, ("GITLAB_GET_PROJECTS", {}))

    if dry_run:
        skip(f"Would execute {action} via registry — dry run")
    else:
        result = await registry.execute("composio", action, params)
        if result.success:
            ok(f"Registry.execute(composio, {action}) → success")
        else:
            fail(f"Registry.execute(composio, {action}) → {result.error}")

    # Disconnect
    await registry.disconnect_all()
    ok("Registry.disconnect_all() complete")
    IntegrationRegistry.reset()


async def test_text_protocol(
    toolkits_filter: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Test 5: ACTION: text-protocol extraction + dispatch (E2E)."""
    header("Test 5: ACTION: Text Protocol E2E")

    from aifactory.integrations.executor import (
        execute_integration_actions,
        extract_action_blocks,
    )
    from aifactory.integrations.providers.composio import ComposioProvider
    from aifactory.integrations.registry import IntegrationRegistry

    IntegrationRegistry.reset()
    registry = IntegrationRegistry()
    provider = ComposioProvider()
    registry.register(provider)
    await registry.connect_all()

    # Simulate what a worker would output
    tk = (toolkits_filter or ["github"])[0]
    action, params = SAFE_ACTIONS.get(tk, ("GITLAB_GET_PROJECTS", {}))

    worker_output = (
        "I'll check the repos for you.\n\n"
        f"ACTION:composio\n"
        f"{json.dumps({'tool': action, 'params': params})}\n"
        f"ACTION_END\n\n"
        "Let me know if you need anything else."
    )

    # Phase 1: extract blocks
    blocks = extract_action_blocks(worker_output)
    if len(blocks) == 1:
        ok(f"Extracted 1 ACTION block: {blocks[0][0]}/{blocks[0][1]['tool']}")
    else:
        fail(f"Expected 1 block, got {len(blocks)}")

    if dry_run:
        skip("Would dispatch to Composio — dry run")
    else:
        # Phase 2: full dispatch
        t0 = time.perf_counter()
        results = await execute_integration_actions(worker_output, registry=registry)
        dt = time.perf_counter() - t0

        if results and results[0][1].success:
            ok(f"Text protocol dispatch → success in {dt:.2f}s")
            info(f"  Tool: {results[0][0]}")
            info(f"  Output: {results[0][1].output[:200]}")
        elif results:
            fail(f"Text protocol dispatch → {results[0][1].error} ({dt:.2f}s)")
        else:
            fail("Text protocol dispatch returned no results")

    await registry.disconnect_all()
    IntegrationRegistry.reset()


async def test_api_endpoints() -> None:
    """Test 6: API endpoint smoke test (optional — requires server running)."""
    header("Test 6: API Endpoint Smoke Test")

    try:
        import httpx
    except ImportError:
        skip("httpx not installed — skipping API tests")
        return

    base = os.getenv("AIFACTORY_API_URL", "http://localhost:8000")

    async with httpx.AsyncClient(timeout=10) as client:
        # GET /api/integrations
        try:
            resp = await client.get(f"{base}/api/integrations")
            if resp.status_code == 200:
                data = resp.json()
                ok(
                    f"GET /api/integrations → {resp.status_code} "
                    f"({data.get('summary', {}).get('total', 0)} integrations)"
                )
            else:
                fail(f"GET /api/integrations → {resp.status_code}")
        except httpx.ConnectError:
            skip(f"API server not running at {base} — skipping endpoint tests")
            return
        except Exception as e:
            fail(f"GET /api/integrations → {e}")
            return

        # POST /api/integrations/composio/test
        try:
            resp = await client.post(f"{base}/api/integrations/composio/test")
            if resp.status_code == 200:
                data = resp.json()
                ok(f"POST /api/integrations/composio/test → {data.get('status')}")
            else:
                fail(f"POST /api/integrations/composio/test → {resp.status_code}")
        except Exception as e:
            fail(f"POST /api/integrations/composio/test → {e}")


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv

    toolkits_filter = args if args else None

    print(f"\n{BOLD}═══ AI Factory Integration — Live Test ═══{RESET}")
    print(f"{DIM}Toolkits: {', '.join(toolkits_filter) if toolkits_filter else 'all enabled'}")
    mode = "DRY RUN (no actions executed)" if dry_run else "LIVE (will call real APIs)"
    print(f"Mode: {mode}{RESET}")

    # ── Test 1: Provider lifecycle ────────────────────────────────────
    provider = await test_provider_lifecycle(dry_run=dry_run)
    if not provider:
        print(f"\n{RED}Cannot proceed without Composio connection.{RESET}")
        print(f"{YELLOW}Setup:{RESET}")
        print("  1. echo 'COMPOSIO_API_KEY=your_key' >> .env")
        print("  2. uv sync --extra integrations")
        print("  3. uv run composio add github")
        _summary()
        return

    # ── Test 2: Tool discovery ────────────────────────────────────────
    tools = await test_tool_discovery(provider, toolkits_filter)

    # ── Test 3: Safe actions ──────────────────────────────────────────
    await test_safe_actions(provider, tools, toolkits_filter, dry_run)

    # Disconnect provider before registry tests (which create their own)
    await provider.disconnect()

    # ── Test 4: Registry flow ─────────────────────────────────────────
    await test_registry_flow(toolkits_filter, dry_run)

    # ── Test 5: Text protocol E2E ─────────────────────────────────────
    await test_text_protocol(toolkits_filter, dry_run)

    # ── Test 6: API endpoints ─────────────────────────────────────────
    await test_api_endpoints()

    _summary()


def _summary() -> None:
    total = passed + failed + skipped
    print(f"\n{BOLD}{'═' * 64}")
    print(
        f"  Results: {GREEN}{passed} passed{RESET}, "
        f"{RED if failed else DIM}{failed} failed{RESET}, "
        f"{YELLOW if skipped else DIM}{skipped} skipped{RESET} "
        f"{DIM}({total} total){RESET}"
    )
    print(f"{BOLD}{'═' * 64}{RESET}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
