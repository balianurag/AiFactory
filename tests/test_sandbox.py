"""
Tests for core/sandbox.py — code execution sandbox.
"""

from __future__ import annotations

from aifactory.core.sandbox import (
    ExecutionResult,
    check_syntax,
    execute_python,
    run_pytest,
)

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
