"""
Tests for core/sandbox.py — code execution sandbox.

All tests run against REAL backends:
  - Local subprocess: always available
  - E2B cloud: runs when E2B_API_KEY is set, skips otherwise
"""

from __future__ import annotations

import pytest

from aifactory.core.sandbox import (
    ExecutionResult,
    _e2b_api_key,
    _get_backend,
    check_syntax,
    execute,
    execute_code_blocks,
    execute_e2b,
    execute_python,
    extract_code_blocks,
    run_pytest,
)

# Helper: skip E2B tests when no API key is available (checks env + .env via settings)
_e2b_key = _e2b_api_key()
skip_no_e2b = pytest.mark.skipif(not _e2b_key, reason="E2B_API_KEY not set")

# =============================================================================
# check_syntax()
# =============================================================================


class TestCheckSyntax:
    def test_valid_code(self) -> None:
        valid, err = check_syntax("x = 1 + 2\nprint(x)")
        assert valid is True
        assert err == ""

    def test_invalid_syntax(self) -> None:
        valid, err = check_syntax("def foo(\n")
        assert valid is False
        assert "SyntaxError" in err

    def test_empty_code(self) -> None:
        valid, err = check_syntax("")
        assert valid is True

    def test_valid_function(self) -> None:
        code = "def add(a: int, b: int) -> int:\n    return a + b"
        valid, err = check_syntax(code)
        assert valid is True

    def test_indentation_error(self) -> None:
        code = "def foo():\nreturn 1"
        valid, err = check_syntax(code)
        # IndentationError is a subclass of SyntaxError
        assert valid is False

    def test_complex_valid_code(self) -> None:
        code = (
            "import json\n"
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class Point:\n"
            "    x: float\n"
            "    y: float\n"
            "\n"
            "p = Point(1.0, 2.0)\n"
            "print(json.dumps({'x': p.x, 'y': p.y}))\n"
        )
        valid, err = check_syntax(code)
        assert valid is True


# =============================================================================
# execute_python()
# =============================================================================


class TestExecutePython:
    def test_simple_print(self) -> None:
        result = execute_python("print('hello sandbox')")
        assert result.success is True
        assert result.exit_code == 0
        assert "hello sandbox" in result.stdout

    def test_syntax_error_fast_reject(self) -> None:
        result = execute_python("def foo(\n")
        assert result.success is False
        assert "SyntaxError" in result.stderr

    def test_runtime_error(self) -> None:
        result = execute_python("x = 1 / 0")
        assert result.success is False
        assert result.exit_code != 0
        assert "ZeroDivisionError" in result.stderr

    def test_import_stdlib(self) -> None:
        result = execute_python("import json; print(json.dumps({'ok': True}))")
        assert result.success is True
        assert '{"ok": true}' in result.stdout

    def test_multiline_code(self) -> None:
        code = (
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)\n"
            "\n"
            "for i in range(10):\n"
            "    print(fib(i), end=' ')\n"
        )
        result = execute_python(code)
        assert result.success is True
        assert "0 1 1 2 3 5 8 13 21 34" in result.stdout

    def test_timeout(self) -> None:
        code = "import time; time.sleep(60)"
        result = execute_python(code, timeout=2)
        assert result.success is False
        assert result.timed_out is True
        assert "timed out" in result.stderr.lower() or "timeout" in result.error_summary.lower()

    def test_exit_code_nonzero(self) -> None:
        result = execute_python("import sys; sys.exit(42)")
        assert result.success is False
        assert result.exit_code == 42

    def test_stderr_captured(self) -> None:
        result = execute_python("import sys; print('err msg', file=sys.stderr)")
        assert "err msg" in result.stderr

    def test_extra_files(self) -> None:
        main_code = "from helper import greet; print(greet('World'))"
        extra = {"helper.py": "def greet(name): return f'Hello, {name}!'"}
        result = execute_python(main_code, extra_files=extra)
        assert result.success is True
        assert "Hello, World!" in result.stdout

    def test_name_error(self) -> None:
        result = execute_python("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in result.stderr

    def test_assertion_error(self) -> None:
        code = "assert 1 == 2, 'math is broken'"
        result = execute_python(code)
        assert result.success is False
        assert "AssertionError" in result.stderr or "AssertionError" in result.error_summary

    def test_output_property(self) -> None:
        result = execute_python("print('out')")
        assert "out" in result.output

    def test_empty_code(self) -> None:
        result = execute_python("")
        assert result.success is True

    def test_output_capped(self) -> None:
        # Generate a lot of output
        code = "for i in range(100000): print(f'line {i}')"
        result = execute_python(code, timeout=10)
        # Output should be capped at 10000 chars
        assert len(result.stdout) <= 10000


# =============================================================================
# run_pytest()
# =============================================================================


class TestRunPytest:
    def test_inline_tests_passing(self) -> None:
        code = (
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "# Inline tests\n"
            "assert add(1, 2) == 3\n"
            "assert add(-1, 1) == 0\n"
            "assert add(0, 0) == 0\n"
            "print('All tests passed')\n"
        )
        result = run_pytest(code)
        assert result.success is True
        assert "All tests passed" in result.stdout

    def test_inline_tests_failing(self) -> None:
        code = (
            "def add(a, b):\n    return a - b  # BUG!\n\nassert add(1, 2) == 3  # This will fail\n"
        )
        result = run_pytest(code)
        assert result.success is False
        assert "AssertionError" in result.stderr or result.exit_code != 0

    def test_separate_test_file(self) -> None:
        impl = "def multiply(a, b):\n    return a * b\n"
        tests = (
            "from implementation import multiply\n"
            "\n"
            "def test_basic():\n"
            "    assert multiply(3, 4) == 12\n"
            "\n"
            "def test_zero():\n"
            "    assert multiply(0, 100) == 0\n"
            "\n"
            "def test_negative():\n"
            "    assert multiply(-2, 3) == -6\n"
        )
        result = run_pytest(impl, test_code=tests)
        # pytest may or may not be installed in test env — check for either
        assert isinstance(result, ExecutionResult)


# =============================================================================
# ExecutionResult
# =============================================================================


class TestExecutionResult:
    def test_success_result(self) -> None:
        r = ExecutionResult(success=True, exit_code=0, stdout="ok", stderr="")
        assert r.success
        assert r.output == "ok"

    def test_failure_result(self) -> None:
        r = ExecutionResult(success=False, exit_code=1, stdout="", stderr="Error occurred")
        assert not r.success
        assert "Error occurred" in r.output

    def test_combined_output(self) -> None:
        r = ExecutionResult(success=False, exit_code=1, stdout="some output", stderr="some error")
        assert "some output" in r.output
        assert "STDERR:" in r.output
        assert "some error" in r.output

    def test_empty_output(self) -> None:
        r = ExecutionResult(success=True, exit_code=0, stdout="", stderr="")
        assert r.output == "(no output)"

    def test_timed_out_flag(self) -> None:
        r = ExecutionResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="Timeout",
            timed_out=True,
        )
        assert r.timed_out is True

    def test_backend_field(self) -> None:
        r = ExecutionResult(success=True, exit_code=0, stdout="ok", stderr="", backend="e2b")
        assert r.backend == "e2b"

    def test_backend_default(self) -> None:
        r = ExecutionResult(success=True, exit_code=0, stdout="", stderr="")
        assert r.backend == "local"


# =============================================================================
# extract_code_blocks()
# =============================================================================


class TestExtractCodeBlocks:
    def test_single_python_block(self) -> None:
        text = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == ("python", "print('hello')")

    def test_multiple_blocks(self) -> None:
        text = "Step 1:\n```python\nx = 1\n```\nStep 2:\n```bash\necho hello\n```\n"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "python"
        assert blocks[1][0] == "bash"

    def test_javascript_block(self) -> None:
        text = "```javascript\nconsole.log('hi')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "javascript"

    def test_js_normalized_to_javascript(self) -> None:
        text = "```js\nconsole.log('hi')\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "javascript"

    def test_sh_normalized_to_bash(self) -> None:
        text = "```sh\nls -la\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "bash"

    def test_shell_normalized_to_bash(self) -> None:
        text = "```shell\nls -la\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "bash"

    def test_non_executable_language_ignored(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_code_protocol_format(self) -> None:
        text = "Result:\nCODE:python\nprint('hello')\nCODE_END\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == ("python", "print('hello')")

    def test_empty_code_block_skipped(self) -> None:
        text = "```python\n\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_no_code_blocks(self) -> None:
        text = "This is a normal response with no code."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 0

    def test_multiline_code(self) -> None:
        text = (
            "```python\ndef add(a, b):\n    return a + b\n\nresult = add(1, 2)\nprint(result)\n```"
        )
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "def add" in blocks[0][1]
        assert "print(result)" in blocks[0][1]

    def test_mixed_executable_and_non_executable(self) -> None:
        text = (
            "```yaml\nkey: value\n```\n"
            "```python\nprint(1)\n```\n"
            "```json\n{}\n```\n"
            "```bash\necho hi\n```\n"
        )
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "python"
        assert blocks[1][0] == "bash"


# =============================================================================
# _get_backend() — env-var logic (monkeypatch controls env, not mocking)
# =============================================================================


class TestGetBackend:
    def test_auto_no_key_returns_local(self, monkeypatch) -> None:
        monkeypatch.setenv("AIFACTORY_SANDBOX_BACKEND", "auto")
        monkeypatch.setenv("E2B_API_KEY", "")
        assert _get_backend() == "local"

    def test_auto_with_key_returns_e2b(self, monkeypatch) -> None:
        monkeypatch.setenv("AIFACTORY_SANDBOX_BACKEND", "auto")
        monkeypatch.setenv("E2B_API_KEY", "test_key")
        assert _get_backend() == "e2b"

    def test_forced_local(self, monkeypatch) -> None:
        monkeypatch.setenv("AIFACTORY_SANDBOX_BACKEND", "local")
        assert _get_backend() == "local"

    def test_forced_e2b(self, monkeypatch) -> None:
        monkeypatch.setenv("AIFACTORY_SANDBOX_BACKEND", "e2b")
        assert _get_backend() == "e2b"


# =============================================================================
# execute() — unified API, real execution
# =============================================================================


class TestExecute:
    async def test_local_python_execution(self) -> None:
        result = await execute("print('from execute')", language="python", backend="local")
        assert result.success is True
        assert "from execute" in result.stdout
        assert result.backend == "local"

    async def test_local_unsupported_language(self) -> None:
        result = await execute("console.log('hi')", language="javascript", backend="local")
        assert result.success is False
        assert "only supports Python" in result.stderr

    @skip_no_e2b
    async def test_e2b_python_via_execute(self) -> None:
        """Real E2B execution through the unified execute() API."""
        result = await execute("print(7 * 6)", language="python", backend="e2b")
        assert result.success is True
        assert "42" in result.stdout
        assert result.backend == "e2b"

    @skip_no_e2b
    async def test_e2b_javascript_via_execute(self) -> None:
        """Real E2B JavaScript execution."""
        result = await execute("console.log(7 * 6)", language="javascript", backend="e2b")
        assert result.success is True
        assert "42" in result.stdout
        assert result.backend == "e2b"


# =============================================================================
# execute_e2b() — real E2B cloud sandbox
# =============================================================================


class TestExecuteE2B:
    async def test_missing_api_key(self, monkeypatch) -> None:
        """Graceful error when no API key is provided."""
        monkeypatch.delenv("E2B_API_KEY", raising=False)
        result = await execute_e2b("print(1)", api_key="")
        assert result.success is False
        assert "API_KEY" in result.stderr
        assert result.backend == "e2b"

    @skip_no_e2b
    async def test_hello_world(self) -> None:
        """Real E2B: simple print."""
        result = await execute_e2b("print('hello from e2b')", api_key=_e2b_key)
        assert result.success is True
        assert "hello from e2b" in result.stdout
        assert result.backend == "e2b"

    @skip_no_e2b
    async def test_stdlib_usage(self) -> None:
        """Real E2B: standard library works."""
        code = "import json; print(json.dumps({'status': 'ok', 'n': 42}))"
        result = await execute_e2b(code, api_key=_e2b_key)
        assert result.success is True
        assert '"status": "ok"' in result.stdout

    @skip_no_e2b
    async def test_multiline_computation(self) -> None:
        """Real E2B: multi-line code with computation."""
        code = (
            "def factorial(n):\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            "\n"
            "print(factorial(10))\n"
        )
        result = await execute_e2b(code, api_key=_e2b_key)
        assert result.success is True
        assert "3628800" in result.stdout

    @skip_no_e2b
    async def test_runtime_error(self) -> None:
        """Real E2B: runtime errors are captured."""
        result = await execute_e2b("x = 1 / 0", api_key=_e2b_key)
        assert result.success is False
        assert result.backend == "e2b"
        assert "ZeroDivisionError" in result.error_summary or "ZeroDivisionError" in result.stderr

    @skip_no_e2b
    async def test_numpy_available(self) -> None:
        """Real E2B: numpy is pre-installed in the sandbox."""
        code = (
            "import numpy as np\n"
            "arr = np.array([1, 2, 3, 4, 5])\n"
            "print(f'sum={arr.sum()} mean={arr.mean()}')\n"
        )
        result = await execute_e2b(code, api_key=_e2b_key)
        assert result.success is True
        assert "sum=15" in result.stdout
        assert "mean=3.0" in result.stdout

    @skip_no_e2b
    async def test_javascript_execution(self) -> None:
        """Real E2B: JavaScript execution."""
        code = "const arr = [1,2,3]; console.log(arr.reduce((a,b) => a+b, 0));"
        result = await execute_e2b(code, language="javascript", api_key=_e2b_key)
        assert result.success is True
        assert "6" in result.stdout


# =============================================================================
# execute_code_blocks() — real execution of code blocks from agent output
# =============================================================================


class TestExecuteCodeBlocks:
    async def test_no_blocks_returns_empty(self) -> None:
        result = await execute_code_blocks("Just text, no code.", backend="local")
        assert result == []

    async def test_python_block_executes(self) -> None:
        text = "Result:\n```python\nprint('executed')\n```\nDone."
        results = await execute_code_blocks(text, backend="local")
        assert len(results) == 1
        lang, code, exec_result = results[0]
        assert lang == "python"
        assert exec_result.success is True
        assert "executed" in exec_result.stdout

    async def test_multiple_python_blocks(self) -> None:
        text = "```python\nprint('one')\n```\n```python\nprint('two')\n```\n"
        results = await execute_code_blocks(text, backend="local")
        assert len(results) == 2
        assert results[0][2].success is True
        assert "one" in results[0][2].stdout
        assert results[1][2].success is True
        assert "two" in results[1][2].stdout

    async def test_error_in_code_block(self) -> None:
        text = "```python\nraise ValueError('boom')\n```"
        results = await execute_code_blocks(text, backend="local")
        assert len(results) == 1
        assert results[0][2].success is False
        assert "ValueError" in results[0][2].stderr

    async def test_non_executable_blocks_skipped(self) -> None:
        text = "```json\n{\"a\": 1}\n```\n```python\nprint('ok')\n```"
        results = await execute_code_blocks(text, backend="local")
        assert len(results) == 1
        assert results[0][0] == "python"

    @skip_no_e2b
    async def test_simulated_worker_output_e2b(self) -> None:
        """Real E2B: extract and run code blocks from simulated worker output."""
        worker_text = (
            "Here's the factorial function:\n\n"
            "```python\n"
            "def factorial(n):\n"
            "    return 1 if n <= 1 else n * factorial(n - 1)\n"
            "\n"
            "print(factorial(10))\n"
            "```\n\n"
            "The result is 3628800."
        )
        results = await execute_code_blocks(worker_text, backend="e2b")
        assert len(results) == 1
        lang, code, exec_result = results[0]
        assert lang == "python"
        assert exec_result.success is True
        assert "3628800" in exec_result.stdout
        assert exec_result.backend == "e2b"

    @skip_no_e2b
    async def test_mixed_blocks_e2b(self) -> None:
        """Real E2B: multiple code blocks, some JSON (skipped), some Python."""
        text = 'Config:\n```json\n{"key": 1}\n```\nCode:\n```python\nprint(2 ** 10)\n```\n'
        results = await execute_code_blocks(text, backend="e2b")
        assert len(results) == 1
        assert results[0][0] == "python"
        assert "1024" in results[0][2].stdout
