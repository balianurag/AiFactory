"""
Shared fixtures for the AiFactory test suite.

Provides:
  - In-memory Qdrant MemoryRouter
  - Pre-configured AgentConfigs for each role
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from aifactory.config.settings import Settings, get_settings
from aifactory.core.memory_router import MemoryRouter
from aifactory.core.schemas import (
    AgentConfig,
    AgentRole,
    WorkerSpecialty,
)

# ─── Settings ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Clear the lru_cache on get_settings before each test."""
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Return a fresh Settings instance for testing."""
    get_settings.cache_clear()
    return get_settings()


# ─── Memory ───────────────────────────────────────────────────────────


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create a fresh in-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture
def memory(qdrant_client: QdrantClient) -> MemoryRouter:
    """Create an in-memory MemoryRouter."""
    return MemoryRouter(qdrant_client=qdrant_client)


# ─── Agent Configs ────────────────────────────────────────────────────


@pytest.fixture
def worker_config() -> AgentConfig:
    """Config for a coder worker."""
    return AgentConfig(
        agent_id="worker_test_001",
        role=AgentRole.WORKER,
        name="Test-Worker-Coder",
        model="openai/gpt-4o-mini",
        specialty=WorkerSpecialty.CODER,
        team_id="team_test",
        supervisor_id="tl_test_001",
        max_iterations=10,
        token_budget=8000,
        temperature=0.2,
    )


@pytest.fixture
def researcher_config() -> AgentConfig:
    """Config for a researcher worker."""
    return AgentConfig(
        agent_id="worker_test_002",
        role=AgentRole.WORKER,
        name="Test-Worker-Researcher",
        model="openai/gpt-4o-mini",
        specialty=WorkerSpecialty.RESEARCHER,
        team_id="team_test",
        supervisor_id="tl_test_001",
        max_iterations=10,
        token_budget=8000,
        temperature=0.2,
    )


@pytest.fixture
def tl_config() -> AgentConfig:
    """Config for a team lead."""
    return AgentConfig(
        agent_id="tl_test_001",
        role=AgentRole.TEAM_LEAD,
        name="Test-Team-Lead",
        model="openai/gpt-4o-mini",
        team_id="team_test",
        supervisor_id="pm_test_001",
        max_iterations=10,
        token_budget=12000,
        temperature=0.3,
    )


@pytest.fixture
def pm_config() -> AgentConfig:
    """Config for a project manager."""
    return AgentConfig(
        agent_id="pm_test_001",
        role=AgentRole.PROJECT_MANAGER,
        name="Test-PM",
        model="openai/gpt-4o-mini",
        max_iterations=8,
        token_budget=16000,
        temperature=0.3,
    )
