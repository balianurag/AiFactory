"""Tests for FILE_READ signal — CodeSecurityProcessor.read_file_safe()."""

from __future__ import annotations

from pathlib import Path

import pytest

from aifactory.agents.code_security_processor import CodeSecurityProcessor


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal fake project with a few source files."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "utils.py").write_text(
        "import os\nimport sys\n\ndef add(a: int, b: int) -> int:\n    return a + b\n\n"
        "class MathHelper:\n    pass\n"
    )
    (src / "secrets.py").write_text(
        'API_KEY = "sk-abc123"\n'
        'password = "hunter2"\n'
        "def connect(): pass\n"
    )
    (src / "large.py").write_text("x = 1\n" * 90_000)  # ~540KB, over 512KB limit
    (tmp_path / "README.md").write_text("# Project\n")
    return tmp_path


class TestReadFileSafe:
    def test_reads_python_file(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/utils.py", project_root=tmp_project)
        assert "error" not in result
        assert result["lang"] == "python"
        assert "add" in result["functions"]
        assert "MathHelper" in result["classes"]
        assert "def add" in result["content"]

    def test_redacts_secret_lines(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/secrets.py", project_root=tmp_project)
        assert "error" not in result
        assert "sk-abc123" not in result["content"]
        assert "hunter2" not in result["content"]
        assert result["secrets_redacted"] == 2
        assert "[REDACTED" in result["content"]
        # Non-secret function still present
        assert "connect" in result["functions"]

    def test_absolute_path_works(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe(str(tmp_project / "src" / "utils.py"))
        assert "error" not in result
        assert "add" in result["functions"]

    def test_missing_file_returns_error(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/nonexistent.py", project_root=tmp_project)
        assert "error" in result
        assert "not found" in result["error"]

    def test_directory_path_returns_error(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src", project_root=tmp_project)
        assert "error" in result
        assert "not a file" in result["error"]

    def test_oversized_file_returns_error(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/large.py", project_root=tmp_project)
        assert "error" in result
        assert "too large" in result["error"]

    def test_content_truncated_at_max_chars(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/utils.py", project_root=tmp_project, max_chars=20)
        assert result["truncated"] is True
        assert len(result["content"]) <= 20 + 60  # content + truncation notice

    def test_no_truncation_for_small_file(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/utils.py", project_root=tmp_project)
        assert result["truncated"] is False

    def test_size_kb_populated(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/utils.py", project_root=tmp_project)
        assert isinstance(result["size_kb"], float)
        assert result["size_kb"] > 0

    def test_path_field_uses_original_input(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/utils.py", project_root=tmp_project)
        assert result["path"] == "src/utils.py"

    def test_non_python_file_still_readable(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("README.md", project_root=tmp_project)
        assert "error" not in result
        assert result["lang"] == "unknown"
        assert "# Project" in result["content"]

    def test_zero_secrets_redacted_for_clean_file(self, tmp_project: Path) -> None:
        csp = CodeSecurityProcessor()
        result = csp.read_file_safe("src/utils.py", project_root=tmp_project)
        assert result["secrets_redacted"] == 0
