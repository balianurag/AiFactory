"""
Base classes for all AiFactory integrations.

Every integration must subclass BaseIntegration and implement:
  - connect()      → establish connection / validate credentials
  - disconnect()   → clean up resources
  - health_check() → verify the integration is alive

Data source integrations additionally implement:
  - ingest()       → pull documents for RAG indexing
  - search()       → query the source directly (optional)

Integration actions are dispatched via execute_action() and return ActionResult.
The pattern mirrors sandbox.py's ExecutionResult for consistency.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class IntegrationStatus(StrEnum):
    """Lifecycle state of an integration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    DISABLED = "disabled"


class IntegrationCategory(StrEnum):
    """High-level grouping."""

    COLLABORATION = "collaboration"  # Slack, Teams, Zoom
    DEVOPS = "devops"  # GitHub, Jira, GitLab, Linear
    DATA_SOURCE = "data_source"  # S3, GCS, PostgreSQL, Drive
    COMMUNICATION = "communication"  # Email, Calendar
    PRODUCTIVITY = "productivity"  # Notion, Confluence
    SEARCH = "search"  # Google, Tavily, Perplexity


class IntegrationCapability(StrEnum):
    """What an integration can do — used for agent tool routing."""

    READ = "read"  # Pull data / fetch information
    WRITE = "write"  # Create / update records
    SEARCH = "search"  # Query / semantic search
    NOTIFY = "notify"  # Send notifications / messages
    INGEST = "ingest"  # Bulk-ingest documents for RAG
    EXECUTE = "execute"  # Trigger actions (create ticket, merge PR, etc.)


# =============================================================================
# Configuration
# =============================================================================


class IntegrationConfig(BaseModel):
    """Declarative config for an integration instance — loadable from YAML."""

    name: str = Field(description="Unique integration name, e.g. 'github', 'composio'")
    display_name: str = Field(default="", description="Human-friendly label")
    category: IntegrationCategory = IntegrationCategory.DEVOPS
    enabled: bool = Field(default=True)
    capabilities: list[IntegrationCapability] = Field(default_factory=list)

    # Auth — populated from env vars at runtime
    auth_type: str = Field(
        default="api_key",
        description="'token' | 'oauth2' | 'api_key' | 'basic' | 'service_account'",
    )
    auth_env_var: str = Field(
        default="",
        description="Env var name holding the primary credential (e.g. COMPOSIO_API_KEY)",
    )
    base_url: str = Field(default="", description="API base URL (if configurable)")

    # Rate limiting
    rate_limit_rpm: int = Field(default=60, description="Requests per minute")

    # Composio-specific (used by ComposioProvider)
    enabled_toolkits: list[str] = Field(
        default_factory=list,
        description="Composio toolkits to enable, e.g. ['github', 'slack', 'jira']",
    )
    default_user_id: str = Field(
        default="aifactory-deployer",
        description="Default Composio user_id for tool execution",
    )

    # RAG / data source specific
    sync_interval_minutes: int = Field(
        default=60,
        description="How often to re-sync data for RAG (data sources only)",
    )
    chunk_size: int = Field(default=1000, description="Text chunk size for RAG ingestion")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")

    # Arbitrary extra config
    extra: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Action result (mirrors sandbox.py ExecutionResult)
# =============================================================================


class ActionResult(BaseModel):
    """Result of executing an action via an integration."""

    success: bool
    action: str = Field(default="", description="What was done, e.g. 'GITHUB_CREATE_ISSUE'")
    integration: str = Field(default="", description="Which integration performed it")
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = Field(default="")
    url: str = Field(default="", description="Link to the created/updated resource")

    @property
    def output(self) -> str:
        """Human-readable result for injection into worker output."""
        if self.success:
            parts = [f"✓ {self.action}"]
            if self.url:
                parts.append(f"  URL: {self.url}")
            if self.data:
                # Compact summary — avoid flooding the context
                summary_keys = list(self.data.keys())[:5]
                for k in summary_keys:
                    v = self.data[k]
                    v_str = str(v)[:200]
                    parts.append(f"  {k}: {v_str}")
            return "\n".join(parts)
        return f"✗ {self.action}: {self.error}"


# =============================================================================
# Document schema (for data source ingestion)
# =============================================================================


class Document(BaseModel):
    """A document ingested from a data source — ready for RAG indexing."""

    id: str = Field(description="Unique document ID from the source")
    source: str = Field(description="Integration name that produced this document")
    title: str = Field(default="")
    content: str = Field(description="Full text content")
    url: str = Field(default="", description="Link back to the source")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# =============================================================================
# Base Integration
# =============================================================================


class BaseIntegration(ABC):
    """
    Abstract base class for all integrations.

    Subclasses must implement connect(), disconnect(), health_check().
    Optionally override execute_action() for EXECUTE capability
    and get_available_actions() to advertise supported actions.
    """

    provider_type: str = "base"  # Override in subclasses: "mcp", "local_shell", etc.
    action_prefix: str = "mcp"  # The ACTION: block prefix for this provider

    def __init__(self, config: IntegrationConfig) -> None:
        self.config = config
        self.status = IntegrationStatus.DISCONNECTED
        self._connected_at: datetime | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def display_name(self) -> str:
        return self.config.display_name or self.config.name.title()

    @property
    def is_connected(self) -> bool:
        return self.status == IntegrationStatus.CONNECTED

    # ── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection / validate credentials.
        Returns True if successful. Sets self.status accordingly.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up resources, close connections."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the integration is alive and credentials are valid."""
        ...

    # ── Actions ──────────────────────────────────────────────────────

    async def execute_action(self, action: str, params: dict[str, Any]) -> ActionResult:
        """
        Execute a named action (e.g. 'GITHUB_CREATE_ISSUE').
        Override in subclasses that support IntegrationCapability.EXECUTE.
        """
        return ActionResult(
            success=False,
            action=action,
            integration=self.name,
            error=f"{self.name} does not support action: {action}",
        )

    def get_available_actions(self) -> list[str]:
        """Return list of action slugs this integration supports."""
        return []

    def get_available_toolkits(self) -> list[str]:
        """Return list of toolkit names (Composio concept)."""
        return self.config.enabled_toolkits

    # ── Helpers ──────────────────────────────────────────────────────

    def _set_connected(self) -> None:
        self.status = IntegrationStatus.CONNECTED
        self._connected_at = datetime.now(UTC)
        logger.info("✅ %s connected", self.display_name)

    def _set_error(self, msg: str) -> None:
        self.status = IntegrationStatus.ERROR
        logger.error("❌ %s error: %s", self.display_name, msg)

    def _set_disconnected(self) -> None:
        self.status = IntegrationStatus.DISCONNECTED
        self._connected_at = None

    def info(self) -> dict[str, Any]:
        """Serialisable status snapshot."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.config.category.value,
            "status": self.status.value,
            "capabilities": [c.value for c in self.config.capabilities],
            "enabled": self.config.enabled,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "actions": self.get_available_actions(),
            "toolkits": self.get_available_toolkits(),
        }


# =============================================================================
# Data Source Integration (specialised base for RAG)
# =============================================================================


class DataSourceIntegration(BaseIntegration):
    """
    Base class for integrations that provide documents for RAG indexing.

    Subclasses must implement ingest() and optionally search().
    """

    @abstractmethod
    async def ingest(
        self,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Pull documents from the source for RAG indexing.

        Args:
            since: Only fetch documents updated after this timestamp.
            limit: Max number of documents to return.
            filters: Source-specific filters (e.g. project, repo, space).

        Returns:
            List of Document objects ready for chunking + embedding.
        """
        ...

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Query the data source directly (optional — not all sources support this).
        """
        return []
