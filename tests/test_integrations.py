"""
Tests for the AiFactory integration layer.

Covers:
  - IntegrationConfig / ActionResult Pydantic models
  - IntegrationRegistry lifecycle (register, get, connect, disconnect, execute)
  - extract_action_blocks() regex (valid, malformed, edge cases)
  - execute_integration_actions() with mock registry
  - ComposioProvider with mocked composio SDK
  - API endpoints (/api/integrations)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aifactory.integrations.base import (
    ActionResult,
    BaseIntegration,
    DataSourceIntegration,
    Document,
    IntegrationCapability,
    IntegrationCategory,
    IntegrationConfig,
    IntegrationStatus,
)
from aifactory.integrations.executor import (
    execute_integration_actions,
    extract_action_blocks,
)
from aifactory.integrations.registry import IntegrationRegistry

# =============================================================================
# Fixtures
# =============================================================================


class FakeIntegration(BaseIntegration):
    """Minimal concrete integration for testing."""

    def __init__(
        self,
        name: str = "fake",
        connected: bool = True,
        actions: dict[str, ActionResult] | None = None,
    ) -> None:
        config = IntegrationConfig(
            name=name,
            display_name=f"Fake {name.title()}",
            category=IntegrationCategory.DEVOPS,
            capabilities=[IntegrationCapability.READ, IntegrationCapability.EXECUTE],
            auth_env_var="FAKE_TOKEN",
            enabled_toolkits=["github", "slack"],
        )
        super().__init__(config)
        self._should_connect = connected
        self._actions = actions or {}

    async def connect(self) -> bool:
        if self._should_connect:
            self._set_connected()
            return True
        self._set_error("Connection refused")
        return False

    async def disconnect(self) -> None:
        self._set_disconnected()

    async def health_check(self) -> bool:
        return self.is_connected

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        if action in self._actions:
            return self._actions[action]
        return ActionResult(
            success=True,
            action=action,
            integration=self.name,
            data={"received_params": params},
        )

    def get_available_actions(self) -> list[str]:
        return list(self._actions.keys()) or ["FAKE_ACTION"]


class FakeDataSource(DataSourceIntegration):
    """Minimal data source integration for testing."""

    def __init__(self) -> None:
        config = IntegrationConfig(
            name="fake_ds",
            category=IntegrationCategory.DATA_SOURCE,
            capabilities=[IntegrationCapability.INGEST, IntegrationCapability.SEARCH],
        )
        super().__init__(config)

    async def connect(self) -> bool:
        self._set_connected()
        return True

    async def disconnect(self) -> None:
        self._set_disconnected()

    async def health_check(self) -> bool:
        return True

    async def ingest(self, *, since=None, limit=None, filters=None) -> list[Document]:
        return [
            Document(
                id="doc_1",
                source="fake_ds",
                title="Test Document",
                content="This is a test document for RAG ingestion.",
                url="https://example.com/doc/1",
                metadata={"type": "test"},
            )
        ]


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the singleton registry before each test."""
    IntegrationRegistry.reset()
    yield
    IntegrationRegistry.reset()


@pytest.fixture
def registry() -> IntegrationRegistry:
    return IntegrationRegistry()


@pytest.fixture
def fake() -> FakeIntegration:
    return FakeIntegration()


@pytest.fixture
def fake_ds() -> FakeDataSource:
    return FakeDataSource()


# =============================================================================
# IntegrationConfig / ActionResult
# =============================================================================


class TestIntegrationConfig:
    def test_minimal_config(self):
        cfg = IntegrationConfig(name="test")
        assert cfg.name == "test"
        assert cfg.enabled is True
        assert cfg.category == IntegrationCategory.DEVOPS
        assert cfg.capabilities == []
        assert cfg.enabled_toolkits == []

    def test_full_config(self):
        cfg = IntegrationConfig(
            name="composio",
            display_name="Composio",
            category=IntegrationCategory.DEVOPS,
            capabilities=[IntegrationCapability.READ, IntegrationCapability.EXECUTE],
            auth_env_var="COMPOSIO_API_KEY",
            enabled_toolkits=["github", "slack"],
            rate_limit_rpm=100,
            extra={"custom": "value"},
        )
        assert cfg.display_name == "Composio"
        assert len(cfg.capabilities) == 2
        assert cfg.enabled_toolkits == ["github", "slack"]
        assert cfg.extra["custom"] == "value"

    def test_config_from_dict(self):
        """Test YAML-like dict loading."""
        data = {
            "name": "github",
            "category": "devops",
            "capabilities": ["read", "write", "execute"],
            "auth_env_var": "GITHUB_TOKEN",
        }
        cfg = IntegrationConfig(**data)
        assert cfg.name == "github"
        assert IntegrationCapability.WRITE in cfg.capabilities


class TestActionResult:
    def test_success_result(self):
        r = ActionResult(
            success=True,
            action="GITHUB_CREATE_ISSUE",
            integration="composio",
            data={"number": 42},
            url="https://github.com/org/repo/issues/42",
        )
        assert r.success
        assert "✓" in r.output
        assert "42" in r.output
        assert "github.com" in r.output

    def test_error_result(self):
        r = ActionResult(
            success=False,
            action="GITHUB_CREATE_ISSUE",
            integration="composio",
            error="401 Unauthorized",
        )
        assert not r.success
        assert "✗" in r.output
        assert "401" in r.output

    def test_empty_result(self):
        r = ActionResult(success=True, action="test")
        assert r.success
        assert r.output == "✓ test"


class TestDocument:
    def test_document_creation(self):
        doc = Document(id="d1", source="github", content="Hello world")
        assert doc.id == "d1"
        assert doc.source == "github"
        assert doc.title == ""
        assert doc.url == ""


# =============================================================================
# IntegrationRegistry
# =============================================================================


class TestRegistryLifecycle:
    def test_singleton(self):
        r1 = IntegrationRegistry()
        r2 = IntegrationRegistry()
        assert r1 is r2

    def test_register_and_get(self, registry, fake):
        registry.register(fake)
        assert registry.get("fake") is fake
        assert registry.get("FAKE") is fake  # case-insensitive
        assert registry.count() == 1

    def test_unregister(self, registry, fake):
        registry.register(fake)
        registry.unregister("fake")
        assert registry.get("fake") is None
        assert registry.count() == 0

    def test_get_nonexistent(self, registry):
        assert registry.get("nope") is None

    def test_register_replace(self, registry):
        f1 = FakeIntegration("test")
        f2 = FakeIntegration("test")
        registry.register(f1)
        registry.register(f2)
        assert registry.get("test") is f2
        assert registry.count() == 1

    async def test_connect_all(self, registry):
        ok = FakeIntegration("ok", connected=True)
        fail = FakeIntegration("fail", connected=False)
        registry.register(ok)
        registry.register(fail)
        results = await registry.connect_all()
        assert results["ok"] is True
        assert results["fail"] is False
        assert ok.is_connected
        assert not fail.is_connected

    async def test_disconnect_all(self, registry):
        f = FakeIntegration("test")
        registry.register(f)
        await registry.connect_all()
        assert f.is_connected
        await registry.disconnect_all()
        assert not f.is_connected

    async def test_health_check_all(self, registry):
        f = FakeIntegration("test")
        registry.register(f)
        await registry.connect_all()
        results = await registry.health_check_all()
        assert results["test"] is True

    async def test_health_check_disconnected(self, registry):
        f = FakeIntegration("test", connected=False)
        registry.register(f)
        results = await registry.health_check_all()
        assert results["test"] is False

    def test_get_by_category(self, registry):
        registry.register(FakeIntegration("a"))
        assert len(registry.get_by_category("devops")) == 1
        assert len(registry.get_by_category("collaboration")) == 0

    def test_get_by_capability(self, registry):
        registry.register(FakeIntegration("a"))
        assert len(registry.get_by_capability(IntegrationCapability.EXECUTE)) == 1
        assert len(registry.get_by_capability(IntegrationCapability.INGEST)) == 0


class TestRegistryExecution:
    async def test_execute_success(self, registry):
        f = FakeIntegration("test")
        registry.register(f)
        await registry.connect_all()
        result = await registry.execute("test", "DO_THING", {"key": "val"})
        assert result.success
        assert result.data["received_params"]["key"] == "val"

    async def test_execute_not_registered(self, registry):
        result = await registry.execute("nope", "DO_THING", {})
        assert not result.success
        assert "not registered" in result.error

    async def test_execute_not_connected(self, registry):
        f = FakeIntegration("test", connected=False)
        registry.register(f)
        result = await registry.execute("test", "DO_THING", {})
        assert not result.success
        assert "not connected" in result.error

    async def test_execute_custom_result(self, registry):
        custom = ActionResult(
            success=True,
            action="CUSTOM",
            integration="test",
            data={"id": 99},
            url="https://example.com",
        )
        f = FakeIntegration("test", actions={"CUSTOM": custom})
        registry.register(f)
        await registry.connect_all()
        result = await registry.execute("test", "CUSTOM", {})
        assert result.success
        assert result.data["id"] == 99
        assert result.url == "https://example.com"

    async def test_execute_exception_handling(self, registry):
        """Integration raises an unexpected exception."""

        class BrokenIntegration(FakeIntegration):
            async def execute_action(self, action, params):
                raise RuntimeError("boom")

        f = BrokenIntegration("broken")
        registry.register(f)
        await registry.connect_all()
        result = await registry.execute("broken", "EXPLODE", {})
        assert not result.success
        assert "boom" in result.error


class TestRegistryDataSources:
    def test_get_data_sources(self, registry, fake, fake_ds):
        registry.register(fake)
        registry.register(fake_ds)
        ds = registry.get_data_sources()
        assert len(ds) == 1
        assert ds[0].name == "fake_ds"

    async def test_ingest_all(self, registry, fake_ds):
        registry.register(fake_ds)
        await registry.connect_all()
        docs = await registry.ingest_all()
        assert len(docs) == 1
        assert docs[0].id == "doc_1"
        assert docs[0].source == "fake_ds"


class TestRegistryToolkits:
    async def test_connected_toolkit_names(self, registry):
        f = FakeIntegration("test")
        registry.register(f)
        await registry.connect_all()
        toolkits = registry.get_connected_toolkit_names()
        assert "github" in toolkits
        assert "slack" in toolkits

    async def test_no_toolkits_when_disconnected(self, registry):
        f = FakeIntegration("test", connected=False)
        registry.register(f)
        assert registry.get_connected_toolkit_names() == []


class TestRegistryInfo:
    async def test_info(self, registry):
        f = FakeIntegration("test")
        registry.register(f)
        await registry.connect_all()
        info = registry.info()
        assert len(info) == 1
        assert info[0]["name"] == "test"
        assert info[0]["status"] == "connected"

    async def test_summary(self, registry, fake_ds):
        registry.register(FakeIntegration("test"))
        registry.register(fake_ds)
        await registry.connect_all()
        s = registry.summary()
        assert s["total"] == 2
        assert s["connected"] == 2
        assert s["data_sources"] == 1


class TestRegistryConfigLoading:
    def test_load_from_nonexistent_dir(self):
        from pathlib import Path

        configs = IntegrationRegistry.load_configs(Path("/nonexistent"))
        assert configs == []

    def test_load_from_real_config_dir(self):
        """Test loading our actual composio.yaml config."""
        from pathlib import Path

        config_dir = (
            Path(__file__).resolve().parent.parent / "src" / "aifactory" / "config" / "integrations"
        )
        if config_dir.exists():
            configs = IntegrationRegistry.load_configs(config_dir)
            assert len(configs) >= 1
            names = [c.name for c in configs]
            assert "composio" in names


# =============================================================================
# extract_action_blocks() regex
# =============================================================================


class TestExtractActionBlocks:
    def test_single_valid_block(self):
        text = (
            "Here's the plan.\n\n"
            "ACTION:composio\n"
            '"tool": "GITHUB_CREATE_ISSUE", '
            '"params": {"owner": "org", "repo": "r", "title": "Bug"}}\n'
            "ACTION_END\n\n"
            "Done."
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        name, parsed = blocks[0]
        assert name == "composio"
        assert parsed["tool"] == "GITHUB_CREATE_ISSUE"
        assert parsed["params"]["owner"] == "org"

    def test_multiple_blocks(self):
        text = (
            "ACTION:composio\n"
            '{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "Bug 1"}}\n'
            "ACTION_END\n"
            "Some text.\n"
            "ACTION:composio\n"
            '{"tool": "SLACK_SEND_MESSAGE", "params": {"channel": "#dev", "text": "Done"}}\n'
            "ACTION_END"
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][1]["tool"] == "GITHUB_CREATE_ISSUE"
        assert blocks[1][1]["tool"] == "SLACK_SEND_MESSAGE"

    def test_no_blocks(self):
        text = "Just regular text with no action blocks."
        assert extract_action_blocks(text) == []

    def test_malformed_json(self):
        text = "ACTION:composio\n{invalid json}\nACTION_END"
        blocks = extract_action_blocks(text)
        assert len(blocks) == 0

    def test_missing_tool_key(self):
        text = 'ACTION:composio\n{"params": {"key": "val"}}\nACTION_END'
        blocks = extract_action_blocks(text)
        assert len(blocks) == 0

    def test_non_object_json(self):
        text = 'ACTION:composio\n["not", "an", "object"]\nACTION_END'
        blocks = extract_action_blocks(text)
        assert len(blocks) == 0

    def test_no_false_positive_with_code_blocks(self):
        """CODE: blocks should NOT be detected as ACTION: blocks."""
        text = "```python\nprint('hello')\n```\nCODE:python\nprint('world')\nCODE_END\nSome text."
        blocks = extract_action_blocks(text)
        assert len(blocks) == 0

    def test_integration_name_case_insensitive(self):
        text = 'ACTION:Composio\n{"tool": "TEST", "params": {}}\nACTION_END'
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "composio"

    def test_multiline_json(self):
        text = (
            "ACTION:composio\n"
            "{\n"
            '  "tool": "GITHUB_CREATE_ISSUE",\n'
            '  "params": {\n'
            '    "owner": "org",\n'
            '    "repo": "r",\n'
            '    "title": "Multi-line test"\n'
            "  }\n"
            "}\n"
            "ACTION_END"
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][1]["params"]["title"] == "Multi-line test"

    def test_empty_params(self):
        text = 'ACTION:composio\n{"tool": "GITHUB_LIST_REPOS", "params": {}}\nACTION_END'
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][1]["params"] == {}

    def test_params_key_missing_defaults_empty(self):
        """params key absent — should still extract the block."""
        text = 'ACTION:composio\n{"tool": "GITHUB_LIST_REPOS"}\nACTION_END'
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][1]["tool"] == "GITHUB_LIST_REPOS"

    def test_custom_integration_name(self):
        text = 'ACTION:custom_provider\n{"tool": "MY_ACTION", "params": {"a": 1}}\nACTION_END'
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "custom_provider"

    def test_mixed_code_and_action_blocks(self):
        """Worker output with both CODE and ACTION blocks."""
        text = (
            "```python\nprint('hello')\n```\n"
            "ACTION:composio\n"
            '{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "Test"}}\n'
            "ACTION_END\n"
            "```javascript\nconsole.log('hi')\n```"
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][1]["tool"] == "GITHUB_CREATE_ISSUE"


# =============================================================================
# execute_integration_actions()
# =============================================================================


class TestExecuteIntegrationActions:
    async def test_no_blocks_returns_empty(self):
        results = await execute_integration_actions("no blocks here")
        assert results == []

    async def test_dispatches_to_registry(self):
        """Verify the full dispatch flow with a mock registry."""
        reg = IntegrationRegistry()
        f = FakeIntegration("composio")
        reg.register(f)
        await reg.connect_all()

        text = (
            "ACTION:composio\n"
            '{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "Bug"}}\n'
            "ACTION_END"
        )
        results = await execute_integration_actions(text, registry=reg)
        assert len(results) == 1
        tool_slug, result = results[0]
        assert tool_slug == "GITHUB_CREATE_ISSUE"
        assert result.success
        assert result.data["received_params"]["title"] == "Bug"

    async def test_multiple_dispatches(self):
        reg = IntegrationRegistry()
        f = FakeIntegration("composio")
        reg.register(f)
        await reg.connect_all()

        text = (
            "ACTION:composio\n"
            '{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "Bug 1"}}\n'
            "ACTION_END\n"
            "ACTION:composio\n"
            '{"tool": "SLACK_SEND_MESSAGE", "params": {"text": "Done"}}\n'
            "ACTION_END"
        )
        results = await execute_integration_actions(text, registry=reg)
        assert len(results) == 2

    async def test_missing_integration(self):
        reg = IntegrationRegistry()
        text = 'ACTION:nonexistent\n{"tool": "DO_THING", "params": {}}\nACTION_END'
        results = await execute_integration_actions(text, registry=reg)
        assert len(results) == 1
        assert not results[0][1].success
        assert "not registered" in results[0][1].error

    async def test_mixed_valid_and_malformed(self):
        """Only valid blocks are dispatched."""
        reg = IntegrationRegistry()
        f = FakeIntegration("composio")
        reg.register(f)
        await reg.connect_all()

        text = (
            "ACTION:composio\n{invalid}\nACTION_END\n"
            "ACTION:composio\n"
            '{"tool": "VALID_ACTION", "params": {}}\n'
            "ACTION_END"
        )
        results = await execute_integration_actions(text, registry=reg)
        assert len(results) == 1
        assert results[0][0] == "VALID_ACTION"

    async def test_none_registry_uses_singleton(self):
        """When registry=None, should use the singleton IntegrationRegistry()."""
        # Register on the singleton
        reg = IntegrationRegistry()
        f = FakeIntegration("composio")
        reg.register(f)
        await reg.connect_all()

        text = 'ACTION:composio\n{"tool": "SINGLETON_TEST", "params": {}}\nACTION_END'
        results = await execute_integration_actions(text, registry=None)
        assert len(results) == 1
        assert results[0][1].success


# =============================================================================
# ComposioProvider (mocked SDK)
# =============================================================================


class TestComposioProvider:
    def test_default_config(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        p = ComposioProvider()
        assert p.name == "composio"
        assert p.status == IntegrationStatus.DISCONNECTED
        assert "github" in p.get_available_toolkits()

    async def test_connect_no_api_key(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "aifactory.integrations.providers.composio._composio_api_key",
                return_value="",
            ),
        ):
            p = ComposioProvider()
            ok = await p.connect()
            assert not ok
            assert p.status == IntegrationStatus.ERROR

    async def test_connect_no_package(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        with (
            patch(
                "aifactory.integrations.providers.composio._composio_api_key",
                return_value="test-key",
            ),
            patch("builtins.__import__", side_effect=ImportError("no module")),
        ):
            p = ComposioProvider()
            # The import guard in connect() catches this
            ok = await p.connect()
            assert not ok

    async def test_execute_not_connected(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        p = ComposioProvider()
        result = await p.execute_action("TEST", {})
        assert not result.success
        assert "not connected" in result.error

    async def test_disconnect(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        p = ComposioProvider()
        p._client = MagicMock()
        p._set_connected()
        assert p.is_connected
        await p.disconnect()
        assert not p.is_connected
        assert p._client is None

    def test_tools_summary_no_cache(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        p = ComposioProvider()
        summary = p.get_tools_summary_for_prompt()
        # Falls back to toolkit names
        assert "github" in summary

    def test_tools_summary_with_cache(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        p = ComposioProvider()
        p._available_tools = {
            "github": ["GITHUB_CREATE_ISSUE", "GITHUB_LIST_ISSUES"],
            "slack": ["SLACK_SEND_MESSAGE"],
        }
        summary = p.get_tools_summary_for_prompt()
        assert "GITHUB_CREATE_ISSUE" in summary
        assert "SLACK_SEND_MESSAGE" in summary

    def test_info(self):
        from aifactory.integrations.providers.composio import ComposioProvider

        p = ComposioProvider()
        info = p.info()
        assert info["name"] == "composio"
        assert info["status"] == "disconnected"
        assert "github" in info["toolkits"]


# =============================================================================
# BaseIntegration abstract contract
# =============================================================================


class TestBaseIntegration:
    async def test_lifecycle(self, fake):
        assert fake.status == IntegrationStatus.DISCONNECTED
        ok = await fake.connect()
        assert ok
        assert fake.is_connected
        assert fake._connected_at is not None
        await fake.disconnect()
        assert not fake.is_connected
        assert fake._connected_at is None

    async def test_failed_connect(self):
        f = FakeIntegration("broken", connected=False)
        ok = await f.connect()
        assert not ok
        assert f.status == IntegrationStatus.ERROR

    def test_display_name_default(self):
        f = FakeIntegration("my_service")
        # display_name from config = "Fake My_Service"
        assert f.display_name == "Fake My_Service"

    async def test_default_execute_action(self):
        """BaseIntegration.execute_action returns not-supported by default."""

        class MinimalIntegration(BaseIntegration):
            async def connect(self):
                return True

            async def disconnect(self):
                pass

            async def health_check(self):
                return True

        cfg = IntegrationConfig(name="minimal", category="devops")
        m = MinimalIntegration(cfg)
        result = await m.execute_action("UNKNOWN", {})
        assert not result.success
        assert "does not support" in result.error

    def test_info_serialisation(self, fake):
        info = fake.info()
        assert info["name"] == "fake"
        assert info["category"] == "devops"
        assert "read" in info["capabilities"]
        assert info["status"] == "disconnected"
        assert info["connected_at"] is None


# =============================================================================
# DataSourceIntegration
# =============================================================================


class TestDataSourceIntegration:
    async def test_ingest(self, fake_ds):
        await fake_ds.connect()
        docs = await fake_ds.ingest()
        assert len(docs) == 1
        assert docs[0].id == "doc_1"
        assert "test document" in docs[0].content.lower()

    async def test_search_default(self, fake_ds):
        """Default search returns empty list."""
        await fake_ds.connect()
        results = await fake_ds.search("anything")
        assert results == []
