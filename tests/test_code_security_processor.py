"""
Tests for code leakage prevention:
  - integrations/executor.py: _looks_like_code, _sanitize_params
  - agents/code_security_processor.py: CodeSecurityProcessor, helpers
  - dynamic_pipeline.py: CODE_ANALYSIS: protocol detection (unit-level)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aifactory.agents.code_security_processor import (
    CodeSecurityProcessor,
    _detect_language,
    _extract_generic_metadata,
    _extract_python_metadata,
    _validate_summary,
)

# =============================================================================
# executor.py — param sanitizer
# =============================================================================
from aifactory.integrations.executor import (
    _looks_like_code,
    _sanitize_params,
    extract_action_blocks,
)


class TestLooksLikeCode:
    def test_short_string_never_flagged(self) -> None:
        # Short strings (< _MIN_CODE_LEN=300) are always safe
        assert _looks_like_code("def foo(): return 1") is False

    def test_long_code_blob_detected(self) -> None:
        code = (
            "def hello():\n"
            "    return 'world'\n\n"
            "import os\n"
            "class Foo:\n"
            "    pass\n"
            "return foo\n"
            "const x = 1\n"
            "let y = 2\n"
        ) * 20  # repeat to exceed _MIN_CODE_LEN
        assert _looks_like_code(code) is True

    def test_plain_text_not_flagged(self) -> None:
        text = (
            "This is a perfectly normal business description about creating a GitHub issue "
            "to track the login bug that was reported by the QA team last Thursday. "
            "The issue should include steps to reproduce, expected vs actual behaviour, "
            "and the environment details. Please make sure to add appropriate labels. "
        ) * 4  # make it long enough
        assert _looks_like_code(text) is False

    def test_threshold_exactly_2_signals_not_flagged(self) -> None:
        # Only 2 signals: "def " and "class " (below threshold of 3)
        text = "def foo():\n    pass\n\nclass Bar:\n    pass\n" * 20
        assert _looks_like_code(text) is False

    def test_fenced_code_block_detected(self) -> None:
        blob = (
            "```python\ndef solve(n):\n    return n * 2\n```\nimport sys\nclass X:\n    pass\n"
        ) * 15
        assert _looks_like_code(blob) is True


class TestSanitizeParams:
    def test_clean_params_unchanged(self) -> None:
        params = {"owner": "org", "repo": "repo", "title": "Fix login bug"}
        cleaned, redacted = _sanitize_params(params)
        assert cleaned == params
        assert redacted == []

    def test_code_value_redacted(self) -> None:
        code_blob = (
            "def process(data):\n    import os\n    class Handler:\n        pass\n    return data\n"
        ) * 30  # ensure > _MIN_CODE_LEN
        params = {"title": "bug", "body": code_blob}
        cleaned, redacted = _sanitize_params(params)
        assert "params.body" in redacted
        assert "CODE REDACTED" in cleaned["body"]
        assert cleaned["title"] == "bug"

    def test_nested_dict_sanitized(self) -> None:
        code_blob = ("def x():\n    import os\n    class C: pass\n    return True\n") * 30
        params = {"meta": {"payload": code_blob}}
        cleaned, redacted = _sanitize_params(params)
        assert any("payload" in k for k in redacted)
        assert "CODE REDACTED" in cleaned["meta"]["payload"]

    def test_list_values_sanitized(self) -> None:
        code_blob = ("def x():\n    import os\n    class C: pass\n    return True\n") * 30
        params = {"files": [code_blob, "safe_value"]}
        cleaned, redacted = _sanitize_params(params)
        assert len(redacted) == 1
        assert "CODE REDACTED" in cleaned["files"][0]
        assert cleaned["files"][1] == "safe_value"

    def test_non_string_values_unchanged(self) -> None:
        params = {"count": 42, "flag": True, "data": [1, 2, 3]}
        cleaned, redacted = _sanitize_params(params)
        assert cleaned == params
        assert redacted == []

    def test_redaction_message_mentions_protocol(self) -> None:
        code_blob = ("def x():\n    import os\n    class C: pass\n    return True\n") * 30
        params = {"body": code_blob}
        cleaned, _ = _sanitize_params(params)
        assert "CODE_ANALYSIS:" in cleaned["body"]


# =============================================================================
# executor.py — extract_action_blocks
# =============================================================================


class TestExtractActionBlocks:
    def test_valid_block_parsed(self) -> None:
        text = (
            'ACTION:composio\n{"tool": "GITHUB_CREATE_ISSUE", "params": {"title": "x"}}\nACTION_END'
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "composio"
        assert blocks[0][1]["tool"] == "GITHUB_CREATE_ISSUE"

    def test_missing_tool_key_skipped(self) -> None:
        text = 'ACTION:composio\n{"params": {"title": "x"}}\nACTION_END'
        blocks = extract_action_blocks(text)
        assert blocks == []

    def test_malformed_json_skipped(self) -> None:
        text = "ACTION:composio\n{bad json}\nACTION_END"
        blocks = extract_action_blocks(text)
        assert blocks == []

    def test_multiple_blocks_extracted(self) -> None:
        text = (
            'ACTION:composio\n{"tool": "A", "params": {}}\nACTION_END\n'
            'ACTION:mcp\n{"tool": "B", "params": {}}\nACTION_END'
        )
        blocks = extract_action_blocks(text)
        assert len(blocks) == 2

    def test_no_blocks_returns_empty(self) -> None:
        assert extract_action_blocks("No action blocks here.") == []


# =============================================================================
# CodeSecurityProcessor — helpers
# =============================================================================


class TestDetectLanguage:
    def test_python(self) -> None:
        assert _detect_language("app/main.py") == "python"

    def test_typescript(self) -> None:
        assert _detect_language("src/index.ts") == "typescript"

    def test_go(self) -> None:
        assert _detect_language("cmd/server.go") == "go"

    def test_unknown(self) -> None:
        assert _detect_language("README.md") == "unknown"


class TestExtractPythonMetadata:
    def test_functions_extracted(self) -> None:
        code = "def foo(): pass\ndef bar(): pass\n"
        meta = _extract_python_metadata(code)
        assert "foo" in meta["functions"]
        assert "bar" in meta["functions"]

    def test_classes_extracted(self) -> None:
        code = "class Foo: pass\nclass Bar: pass\n"
        meta = _extract_python_metadata(code)
        assert "Foo" in meta["classes"]
        assert "Bar" in meta["classes"]

    def test_imports_extracted(self) -> None:
        code = "import os\nimport sys\nfrom pathlib import Path\n"
        meta = _extract_python_metadata(code)
        assert "os" in meta["imports"]
        assert "pathlib" in meta["imports"]

    def test_deduplication(self) -> None:
        code = "def foo(): pass\ndef foo(): pass\n"  # duplicate
        meta = _extract_python_metadata(code)
        assert meta["functions"].count("foo") == 1

    def test_cap_at_50(self) -> None:
        code = "\n".join(f"def func_{i}(): pass" for i in range(100))
        meta = _extract_python_metadata(code)
        assert len(meta["functions"]) <= 50

    def test_syntax_error_fallback(self) -> None:
        # Invalid Python — regex fallback should still find function names
        code = "def foo():\ndef bar():\n"  # intentionally broken
        meta = _extract_python_metadata(code)
        # Regex fallback: finds names
        assert "foo" in meta["functions"]

    def test_no_code_in_output(self) -> None:
        code = "def secret_key(): return 'MYSECRET12345'\n"
        meta = _extract_python_metadata(code)
        # Only the name — never the body
        assert "secret_key" in meta["functions"]
        assert "MYSECRET12345" not in str(meta)


class TestExtractGenericMetadata:
    def test_js_functions(self) -> None:
        code = "function fetchUser() {}\nfunction saveData() {}\n"
        meta = _extract_generic_metadata(code)
        assert "fetchUser" in meta["functions"]
        assert "saveData" in meta["functions"]

    def test_c_structs(self) -> None:
        # Generic regex matches "struct TypeName" (C/C++ style), not Go's "type TypeName struct"
        code = "struct Server {};\nstruct Config {};\n"
        meta = _extract_generic_metadata(code)
        assert "Server" in meta["classes"]
        assert "Config" in meta["classes"]


class TestValidateSummary:
    def test_fenced_code_stripped(self) -> None:
        text = "Overview: ```python\ndef foo(): pass\n``` Done."
        result = _validate_summary(text)
        assert "def foo" not in result
        assert "[code removed" in result

    def test_long_inline_code_stripped(self) -> None:
        text = "Use `some_function_with_a_very_long_name(arg1, arg2)` to do it."
        result = _validate_summary(text)
        assert "`some_function" not in result

    def test_clean_text_passes_through(self) -> None:
        text = "The codebase has three main modules: auth, routes, and models."
        assert _validate_summary(text) == text


# =============================================================================
# CodeSecurityProcessor — read_from_path
# =============================================================================


class TestReadFromPath:
    def test_reads_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("def hello(): pass\n")
        codebase = CodeSecurityProcessor.read_from_path(tmp_path)
        assert any("app.py" in k for k in codebase)

    def test_skips_non_code_files(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("def hello(): pass\n")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
        codebase = CodeSecurityProcessor.read_from_path(tmp_path)
        assert all("image.png" not in k for k in codebase)

    def test_skips_pycache(self, tmp_path: Path) -> None:
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "app.cpython-311.pyc").write_text("bytecode")
        (tmp_path / "main.py").write_text("print('hi')\n")
        codebase = CodeSecurityProcessor.read_from_path(tmp_path)
        assert all("__pycache__" not in k for k in codebase)

    def test_skips_git_directory(self, tmp_path: Path) -> None:
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("[core]")
        (tmp_path / "main.py").write_text("print('hi')\n")
        codebase = CodeSecurityProcessor.read_from_path(tmp_path)
        assert all(".git" not in k for k in codebase)

    def test_returns_empty_for_nonexistent_path(self) -> None:
        codebase = CodeSecurityProcessor.read_from_path("/path/that/does/not/exist")
        assert codebase == {}

    def test_max_files_cap(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(f"# file {i}\n")
        codebase = CodeSecurityProcessor.read_from_path(tmp_path, max_files=5)
        assert len(codebase) == 5

    def test_relative_paths_in_output(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "module.py").write_text("x = 1\n")
        codebase = CodeSecurityProcessor.read_from_path(tmp_path)
        # Keys should be relative, not absolute
        assert all(not Path(k).is_absolute() for k in codebase)


# =============================================================================
# CodeSecurityProcessor — process_codebase
# =============================================================================


class TestProcessCodebase:
    @pytest.fixture
    def processor(self) -> CodeSecurityProcessor:
        return CodeSecurityProcessor()

    @pytest.fixture
    def small_codebase(self) -> dict[str, str]:
        return {
            "app.py": "def main():\n    pass\n",
            "utils.py": "def helper():\n    return True\n",
        }

    async def test_empty_codebase_returns_safe_result(
        self, processor: CodeSecurityProcessor
    ) -> None:
        result = await processor.process_codebase({})
        assert result["file_count"] == 0
        assert result["warning"] == "raw_code_never_exposed"

    async def test_result_contains_required_keys(
        self, processor: CodeSecurityProcessor, small_codebase: dict
    ) -> None:
        result = await processor.process_codebase(small_codebase)
        assert "safe_summary" in result
        assert "metadata" in result
        assert "file_count" in result
        assert "total_bytes" in result
        assert "languages" in result
        assert result["warning"] == "raw_code_never_exposed"

    async def test_file_count_correct(
        self, processor: CodeSecurityProcessor, small_codebase: dict
    ) -> None:
        result = await processor.process_codebase(small_codebase)
        assert result["file_count"] == 2

    async def test_metadata_contains_function_names(
        self, processor: CodeSecurityProcessor, small_codebase: dict
    ) -> None:
        result = await processor.process_codebase(small_codebase)
        meta = result["metadata"]
        assert "app.py" in meta
        assert "main" in meta["app.py"]["functions"]

    async def test_no_raw_code_in_metadata(self, processor: CodeSecurityProcessor) -> None:
        codebase = {"secret.py": "API_KEY = 'sk-prod-secret12345'\ndef leak(): return API_KEY\n"}
        result = await processor.process_codebase(codebase)
        # Metadata should only have names, never values
        assert "sk-prod-secret12345" not in json.dumps(result["metadata"])

    async def test_llm_summary_called_when_think_fn_provided(
        self, processor: CodeSecurityProcessor, small_codebase: dict
    ) -> None:
        mock_think = AsyncMock(return_value="The codebase contains two Python modules.")
        result = await processor.process_codebase(small_codebase, llm_think_fn=mock_think)
        assert mock_think.called
        assert result["safe_summary"] != ""

    async def test_no_llm_call_without_think_fn(
        self, processor: CodeSecurityProcessor, small_codebase: dict
    ) -> None:
        result = await processor.process_codebase(small_codebase, llm_think_fn=None)
        assert result["safe_summary"] == ""
        assert result["file_count"] == 2

    async def test_summary_validated_code_stripped(
        self, processor: CodeSecurityProcessor, small_codebase: dict
    ) -> None:
        # LLM accidentally includes a code block in its summary
        mock_think = AsyncMock(
            return_value=(
                "The codebase has a main function.\n"
                "```python\ndef main(): pass\n```\n"
                "It also has helpers."
            )
        )
        result = await processor.process_codebase(small_codebase, llm_think_fn=mock_think)
        assert "```" not in result["safe_summary"]
        assert "[code removed" in result["safe_summary"]

    async def test_languages_detected(self, processor: CodeSecurityProcessor) -> None:
        codebase = {
            "main.py": "def x(): pass\n",
            "index.ts": "const x = 1;\n",
        }
        result = await processor.process_codebase(codebase)
        assert "python" in result["languages"]
        assert "typescript" in result["languages"]


# =============================================================================
# CODE_ANALYSIS: protocol in dynamic_pipeline — unit test via mock
# =============================================================================


class TestCodeAnalysisProtocol:
    """
    Test the CODE_ANALYSIS: dispatch logic in _run_worker_task without
    running the full pipeline.  We replicate the detect-and-re-call pattern
    from the pipeline code in a self-contained test.
    """

    def _build_protocol_response(self, path: str) -> str:
        return f"I need to analyse the codebase.\nCODE_ANALYSIS: {path}\n"

    async def test_code_analysis_signal_detected(self) -> None:
        raw = self._build_protocol_response("/app/project")
        import re

        match = re.search(r"(?m)^CODE_ANALYSIS:\s*(.+)", raw, re.IGNORECASE)
        assert match is not None
        assert match.group(1).strip() == "/app/project"

    async def test_code_analysis_no_signal(self) -> None:
        raw = "I analysed the task and here is the result.\n"
        import re

        match = re.search(r"(?m)^CODE_ANALYSIS:\s*(.+)", raw, re.IGNORECASE)
        assert match is None

    async def test_code_analysis_triggers_processor(self) -> None:
        """Mock the processor to verify it's called with the right path."""
        import re

        mock_processor = MagicMock()
        mock_processor.read_from_path = MagicMock(return_value={"app.py": "def main(): pass\n"})
        mock_processor.process_codebase = AsyncMock(
            return_value={
                "safe_summary": "Single Python module with main function.",
                "metadata": {"app.py": {"functions": ["main"], "classes": []}},
                "file_count": 1,
                "total_bytes": 22,
                "languages": ["python"],
                "warning": "raw_code_never_exposed",
            }
        )

        raw = "I need to look at the code.\nCODE_ANALYSIS: /project/src\n"
        match = re.search(r"(?m)^CODE_ANALYSIS:\s*(.+)", raw, re.IGNORECASE)
        assert match is not None

        path = match.group(1).strip()
        codebase = mock_processor.read_from_path(path)
        safe_data = await mock_processor.process_codebase(codebase)

        mock_processor.read_from_path.assert_called_once_with("/project/src")
        mock_processor.process_codebase.assert_called_once()
        assert safe_data["warning"] == "raw_code_never_exposed"
        assert "raw" not in safe_data["safe_summary"].lower()

    async def test_empty_path_yields_no_files(self) -> None:
        processor = CodeSecurityProcessor()
        codebase = processor.read_from_path("/nonexistent/path/xyz")
        assert codebase == {}

    async def test_safe_data_has_no_raw_code(self) -> None:
        processor = CodeSecurityProcessor()
        # Simulate a codebase with sensitive-looking code
        codebase = {"secret.py": ("API_KEY = 'sk-secret'\ndef get_key():\n    return API_KEY\n")}
        result = await processor.process_codebase(codebase, llm_think_fn=None)
        # Without LLM, summary is empty — metadata should have only names
        assert "sk-secret" not in json.dumps(result)
        assert result["warning"] == "raw_code_never_exposed"
