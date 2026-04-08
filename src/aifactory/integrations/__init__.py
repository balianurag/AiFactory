"""
AiFactory Integrations — Modular connectors for external services and data sources.

Architecture:
  BaseIntegration         → abstract contract all integrations implement
  DataSourceIntegration   → specialised base for RAG data sources
  IntegrationRegistry     → singleton that discovers, loads, and manages integrations
  ComposioProvider        → default backend (1000+ tools via Composio Direct Execution)

Integrations are config-driven (YAML) and auto-discovered at startup.
Workers invoke tools via ACTION: text-protocol blocks in their output.
"""

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
from aifactory.integrations.registry import IntegrationRegistry

__all__ = [
    "ActionResult",
    "BaseIntegration",
    "DataSourceIntegration",
    "Document",
    "IntegrationCapability",
    "IntegrationCategory",
    "IntegrationConfig",
    "IntegrationRegistry",
    "IntegrationStatus",
]
