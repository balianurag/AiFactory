#!/usr/bin/env python3
"""
Live E2B sandbox integration test.

Runs real code through both LOCAL and E2B backends, plus the
execute_code_blocks() pipeline helper that mirrors what workers do.

Usage:
    uv run python scripts/test_e2b_live.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

from aifactory.core.sandbox import (  # noqa: E402
    execute,
    execute_code_blocks,
    execute_e2b,
    execute_python,
    extract_code_blocks,
)

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}{RESET}\n")


def result_box(label: str, res) -> None:
    status = f"{GREEN}✓ SUCCESS{RESET}" if res.success else f"{RED}✗ FAILED{RESET}"
    print(f"  {BOLD}{label}{RESET}  [{res.backend}]  {status}")
    if res.stdout.strip():
        for line in res.stdout.strip().splitlines()[:10]:
            print(f"    stdout: {line}")
    if res.stderr.strip():
        for line in res.stderr.strip().splitlines()[:5]:
            print(f"    {RED}stderr: {line}{RESET}")
    if res.timed_out:
        print(f"    {YELLOW}⏱ Timed out{RESET}")
    print()


async def main() -> None:
    e2b_key = os.getenv("E2B_API_KEY", "")
    print(f"{BOLD}E2B API Key:{RESET} {'set (' + e2b_key[:10] + '...)' if e2b_key else 'NOT SET'}")

    # ── Test 1: LOCAL Python execution ────────────────────────────────
    header("Test 1: Local Python — Hello World")
    t0 = time.perf_counter()
    r = execute_python("print('Hello from local sandbox!')")
    dt = time.perf_counter() - t0
    result_box(f"Local Python ({dt:.2f}s)", r)

    # ── Test 2: LOCAL Python — Fibonacci ──────────────────────────────
    header("Test 2: Local Python — Fibonacci sequence")
    code = """\
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("First 15 Fibonacci numbers:")
print(list(fib(15)))
"""
    t0 = time.perf_counter()
    r = execute_python(code)
    dt = time.perf_counter() - t0
    result_box(f"Local Fibonacci ({dt:.2f}s)", r)

    # ── Test 3: LOCAL Python — Data processing ────────────────────────
    header("Test 3: Local Python — JSON + data processing")
    code = """\
import json
from collections import Counter

data = [
    {"name": "Alice", "score": 95, "grade": "A"},
    {"name": "Bob", "score": 87, "grade": "B"},
    {"name": "Charlie", "score": 92, "grade": "A"},
    {"name": "Diana", "score": 78, "grade": "C"},
    {"name": "Eve", "score": 91, "grade": "A"},
]

avg = sum(d["score"] for d in data) / len(data)
grades = Counter(d["grade"] for d in data)
top = max(data, key=lambda d: d["score"])

result = {
    "average_score": round(avg, 1),
    "grade_distribution": dict(grades),
    "top_student": top["name"],
}
print(json.dumps(result, indent=2))
"""
    t0 = time.perf_counter()
    r = execute_python(code)
    dt = time.perf_counter() - t0
    result_box(f"Local Data Processing ({dt:.2f}s)", r)

    # ── Test 4: LOCAL Python — Error handling ─────────────────────────
    header("Test 4: Local Python — Runtime error")
    t0 = time.perf_counter()
    r = execute_python("x = [1, 2, 3]\nprint(x[10])")
    dt = time.perf_counter() - t0
    result_box(f"Local Error ({dt:.2f}s)", r)

    # ── Test 5: E2B Python execution ─────────────────────────────────
    if not e2b_key:
        print(f"\n{YELLOW}⚠ Skipping E2B tests — E2B_API_KEY not set{RESET}\n")
    else:
        header("Test 5: E2B Cloud — Hello World")
        t0 = time.perf_counter()
        r = await execute_e2b("print('Hello from E2B cloud sandbox!')", api_key=e2b_key)
        dt = time.perf_counter() - t0
        result_box(f"E2B Hello ({dt:.2f}s)", r)

        # ── Test 6: E2B Python — numpy (pre-installed in E2B) ────────
        header("Test 6: E2B Cloud — NumPy computation")
        code = """\
import numpy as np

# Create a random 5x5 matrix
np.random.seed(42)
matrix = np.random.rand(5, 5)
eigenvalues = np.linalg.eigvals(matrix)

print("Matrix shape:", matrix.shape)
print("Mean:", round(matrix.mean(), 4))
print("Eigenvalues:", [round(float(v), 4) for v in eigenvalues])
print("Determinant:", round(float(np.linalg.det(matrix)), 6))
"""
        t0 = time.perf_counter()
        r = await execute_e2b(code, api_key=e2b_key)
        dt = time.perf_counter() - t0
        result_box(f"E2B NumPy ({dt:.2f}s)", r)

        # ── Test 7: E2B JavaScript ───────────────────────────────────
        header("Test 7: E2B Cloud — JavaScript")
        js_code = """\
const data = [10, 20, 30, 40, 50];
const sum = data.reduce((a, b) => a + b, 0);
const avg = sum / data.length;
console.log(`Sum: ${sum}, Average: ${avg}`);
console.log(`Squares: ${data.map(x => x * x)}`);
"""
        t0 = time.perf_counter()
        r = await execute_e2b(js_code, language="javascript", api_key=e2b_key)
        dt = time.perf_counter() - t0
        result_box(f"E2B JavaScript ({dt:.2f}s)", r)

        # ── Test 8: E2B Python — Error handling ──────────────────────
        header("Test 8: E2B Cloud — Runtime error")
        t0 = time.perf_counter()
        r = await execute_e2b("result = 1 / 0", api_key=e2b_key)
        dt = time.perf_counter() - t0
        result_box(f"E2B Error ({dt:.2f}s)", r)

    # ── Test 9: Unified execute() — auto backend ─────────────────────
    header("Test 9: Unified execute() — auto-detect backend")
    t0 = time.perf_counter()
    r = await execute("import sys; print(f'Python {sys.version}')")
    dt = time.perf_counter() - t0
    result_box(f"Unified execute ({dt:.2f}s)", r)

    # ── Test 10: execute_code_blocks() — simulated worker output ─────
    header("Test 10: execute_code_blocks() — simulated worker response")
    worker_output = """\
Here's my analysis of the sorting algorithms:

**Bubble Sort** has O(n²) complexity. Here's a quick benchmark:

```python
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

data = list(range(500, 0, -1))
t0 = time.perf_counter()
bubble_sort(data.copy())
dt = time.perf_counter() - t0
print(f"Bubble sort 500 elements: {dt*1000:.1f}ms")
```

And **Python's built-in sort** (Timsort) is much faster:

```python
import time

data = list(range(500, 0, -1))
t0 = time.perf_counter()
sorted(data)
dt = time.perf_counter() - t0
print(f"Timsort 500 elements: {dt*1000:.4f}ms")
print(f"Timsort is O(n log n)")
```

As you can see, Timsort is significantly faster for this workload.
"""
    # First show extraction
    blocks = extract_code_blocks(worker_output)
    print(f"  Extracted {len(blocks)} code blocks from worker output")
    for i, (lang, code) in enumerate(blocks):
        print(f"    Block {i + 1}: {lang} ({len(code)} chars)")

    # Execute with auto-detect (E2B if key set, else local)
    t0 = time.perf_counter()
    results = await execute_code_blocks(worker_output)
    dt = time.perf_counter() - t0
    print(f"\n  Executed {len(results)} blocks in {dt:.2f}s (auto-backend):")
    for lang, _code, res in results:
        result_box(f"  Block [{lang}]", res)

    # ── Summary ──────────────────────────────────────────────────────
    header("Summary")
    print(f"  {GREEN}All live tests completed!{RESET}")
    print("  Local backend: always available")
    print(f"  E2B backend:   {'available' if e2b_key else 'skipped (no key)'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
