"""
Integration Registry — Discovers, loads, and manages all integrations.

Singleton pattern — one registry per process. Integrations are registered
at startup (from YAML configs or programmatically) and made available to
the pipeline for tool execution.

Usage:
    registry = IntegrationRegistry()
    registry.register(ComposioProvider(config))
    await registry.connect_all()

    # Worker dispatches an action
    result = await registry.execute("composio", "GITHUB_CREATE_ISSUE", {...})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from aifactory.integrations.base import (
    ActionResult,
    BaseIntegration,
    DataSourceIntegration,
    Document,
    IntegrationCapability,
    IntegrationConfig,
)

logger = logging.getLogger(__name__)

# Config directory for integration YAML files
_INTEGRATIONS_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "integrations"


class IntegrationRegistry:
    """
    Central registry for all integrations.

    Responsibilities:
      1. Register integration instances
      2. Load integration configs from YAML
      3. Connect / disconnect all integrations
      4. Provide lookup by name, category, or capability
      5. Execute actions by delegating to the right integration
      6. Aggregate data source ingestion for RAG
    """

    _instance: IntegrationRegistry | None = None

    def __new__(cls) -> IntegrationRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._integrations = {}
            cls._instance._initialized = True
        return cls._instance

    def __init__(self) -> None:
        # __new__ handles singleton init — avoid re-init
        if not hasattr(self, "_integrations"):
            self._integrations: dict[str, BaseIntegration] = {}

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — used in tests."""
        cls._instance = None

    # ── Registration ─────────────────────────────────────────────────

    def register(self, integration: BaseIntegration) -> None:
        """Register an integration instance."""
        name = integration.name.lower()
        if name in self._integrations:
            logger.warning("Integration '%s' already registered — replacing", name)
        self._integrations[name] = integration
        logger.info(
            "Registered integration: %s [%s] capabilities=%s",
            name,
            integration.config.category.value,
            [c.value for c in integration.config.capabilities],
        )

    def unregister(self, name: str) -> None:
        """Remove an integration from the registry."""
        self._integrations.pop(name.lower(), None)

    # ── Lookup ───────────────────────────────────────────────────────

    def get(self, name: str) -> BaseIntegration | None:
        """Get an integration by name."""
        return self._integrations.get(name.lower())

    def get_by_category(self, category: str) -> list[BaseIntegration]:
        """Get all integrations in a category."""
        return [i for i in self._integrations.values() if i.config.category.value == category]

    def get_by_capability(self, capability: IntegrationCapability) -> list[BaseIntegration]:
        """Get all enabled integrations that support a given capability."""
        return [
            i
            for i in self._integrations.values()
            if capability in i.config.capabilities and i.config.enabled
        ]

    def get_data_sources(self) -> list[DataSourceIntegration]:
        """Get all registered data source integrations."""
        return [
            i
            for i in self._integrations.values()
            if isinstance(i, DataSourceIntegration) and i.config.enabled
        ]

    def get_connected(self) -> list[BaseIntegration]:
        """Get all currently connected integrations."""
        return [i for i in self._integrations.values() if i.is_connected]

    def all(self) -> list[BaseIntegration]:
        """Get all registered integrations."""
        return list(self._integrations.values())

    def count(self) -> int:
        return len(self._integrations)

    # ── Action Dispatch ──────────────────────────────────────────────

    async def execute(
        self,
        integration_name: str,
        action: str,
        params: dict[str, Any],
    ) -> ActionResult:
        """
        Execute an action on a named integration.

        This is the primary dispatch method called by execute_integration_actions().
        """
        integration = self.get(integration_name)
        if integration is None:
            return ActionResult(
                success=False,
                action=action,
                integration=integration_name,
                error=f"Integration '{integration_name}' not registered",
            )
        if not integration.is_connected:
            return ActionResult(
                success=False,
                action=action,
                integration=integration_name,
                error=(
                    f"Integration '{integration_name}' is not connected"
                    f" (status: {integration.status})"
                ),
            )
        try:
            return await integration.execute_action(action, params)
        except Exception as e:
            logger.error("Integration action failed: %s/%s: %s", integration_name, action, e)
            return ActionResult(
                success=False,
                action=action,
                integration=integration_name,
                error=str(e),
            )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect_all(self) -> dict[str, bool]:
        """Connect all enabled integrations. Returns {name: success}."""
        results: dict[str, bool] = {}
        for name, integration in self._integrations.items():
            if not integration.config.enabled:
                logger.info("Skipping disabled integration: %s", name)
                results[name] = False
                continue
            try:
                ok = await integration.connect()
                results[name] = ok
            except Exception as e:
                logger.error("Failed to connect %s: %s", name, e)
                integration._set_error(str(e))
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all integrations."""
        for integration in self._integrations.values():
            if integration.is_connected:
                try:
                    await integration.disconnect()
                except Exception as e:
                    logger.error("Error disconnecting %s: %s", integration.name, e)

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all connected integrations."""
        results: dict[str, bool] = {}
        for name, integration in self._integrations.items():
            if integration.is_connected:
                try:
                    results[name] = await integration.health_check()
                except Exception as e:
                    logger.error("Health check failed for %s: %s", name, e)
                    results[name] = False
            else:
                results[name] = False
        return results

    # ── RAG / Data Source Aggregation ────────────────────────────────

    async def ingest_all(
        self,
        *,
        since: Any | None = None,
        limit_per_source: int | None = None,
    ) -> list[Document]:
        """Ingest documents from all connected data sources."""
        all_docs: list[Document] = []
        for ds in self.get_data_sources():
            if not ds.is_connected:
                logger.warning("Skipping disconnected data source: %s", ds.name)
                continue
            try:
                docs = await ds.ingest(since=since, limit=limit_per_source)
                all_docs.extend(docs)
                logger.info("Ingested %d documents from %s", len(docs), ds.name)
            except Exception as e:
                logger.error("Ingestion failed for %s: %s", ds.name, e)
        return all_docs

    # ── YAML Config Loading ──────────────────────────────────────────

    @staticmethod
    def load_configs(config_dir: Path | None = None) -> list[IntegrationConfig]:
        """
        Load integration configs from YAML files.

        Each YAML file defines one integration. Files named after the
        integration: composio.yaml, github.yaml, etc.
        """
        d = config_dir or _INTEGRATIONS_CONFIG_DIR
        if not d.exists():
            logger.debug("No integrations config dir at %s", d)
            return []

        configs: list[IntegrationConfig] = []
        for f in sorted(d.glob("*.yaml")):
            try:
                raw = yaml.safe_load(f.read_text())
                if raw and isinstance(raw, dict):
                    cfg = IntegrationConfig(**raw)
                    configs.append(cfg)
                    logger.debug("Loaded integration config: %s", cfg.name)
            except Exception as e:
                logger.error("Failed to load integration config %s: %s", f.name, e)
        return configs

    # ── Connected Toolkits (for prompt injection) ────────────────────

    def get_connected_toolkit_names(self) -> list[str]:
        """
        Get all toolkit names from connected integrations.

        Used to build the INTEGRATION TOOLS instruction block in worker prompts.
        """
        toolkits: list[str] = []
        for integration in self.get_connected():
            toolkits.extend(integration.get_available_toolkits())
        return sorted(set(toolkits))

    # ── Info / Status ────────────────────────────────────────────────

    def info(self) -> list[dict[str, Any]]:
        """Status snapshot of all integrations."""
        return [i.info() for i in self._integrations.values()]

    def summary(self) -> dict[str, Any]:
        """High-level summary."""
        return {
            "total": self.count(),
            "connected": len(self.get_connected()),
            "data_sources": len(self.get_data_sources()),
            "toolkits": self.get_connected_toolkit_names(),
            "categories": sorted({i.config.category.value for i in self._integrations.values()}),
        }
