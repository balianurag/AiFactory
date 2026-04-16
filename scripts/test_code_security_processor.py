"""
End-to-end smoke test for CodeSecurityProcessor with a real LiteLLM call.

Tests:
  1. read_from_path — scans src/aifactory/core/ for Python files
  2. process_codebase — real LLM generates a high-level summary
  3. Validates the safe output: no raw code, correct keys, summary present
  4. Prints the safe summary so you can visually inspect it

Usage:
    source .venv/bin/activate
    python scripts/test_code_security_processor.py [model]

    model  Optional litellm model string (default: gpt-4o-mini).
           Examples:
             gpt-4o-mini
             anthropic/claude-haiku-4-5-20251001
             openai/gpt-4.1-nano
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main() -> None:
    import litellm

    from aifactory.agents.code_security_processor import CodeSecurityProcessor

    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"

    print(f"Using model: {model}")
    print()

    # ── 1. read files ─────────────────────────────────────────────────────────
    scan_path = Path(__file__).parent.parent / "src" / "aifactory" / "core"
    print(f"Scanning: {scan_path}")

    processor = CodeSecurityProcessor()
    codebase = processor.read_from_path(scan_path, max_files=10)

    print(f"Files read: {len(codebase)}")
    for fp in list(codebase)[:5]:
        print(f"  - {fp} ({len(codebase[fp])} chars)")
    if len(codebase) > 5:
        print(f"  ... and {len(codebase) - 5} more")
    print()

    # ── 2. LLM think function — plain litellm.acompletion, same as BaseAgent ─
    async def think(prompt: str, _ctx: dict) -> str:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    # ── 3. process_codebase ───────────────────────────────────────────────────
    print("Processing codebase (calling LLM for safe summary)...")
    result = await processor.process_codebase(codebase, llm_think_fn=think)

    # ── 4. display ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULT KEYS :", list(result.keys()))
    print(f"file_count  : {result['file_count']}")
    print(f"total_bytes : {result['total_bytes']:,}")
    print(f"languages   : {result['languages']}")
    print(f"warning     : {result['warning']}")
    print()
    print("SAFE SUMMARY:")
    print("-" * 60)
    print(result["safe_summary"])
    print("-" * 60)
    print()

    # ── 5. security assertions ────────────────────────────────────────────────
    print("Running security assertions...")

    summary = result["safe_summary"]
    metadata_str = json.dumps(result["metadata"])
    errors: list[str] = []

    if result.get("warning") != "raw_code_never_exposed":
        errors.append("MISSING SENTINEL: warning != raw_code_never_exposed")

    for key in ("safe_summary", "metadata", "file_count", "total_bytes", "languages", "warning"):
        if key not in result:
            errors.append(f"MISSING KEY: {key}")

    if summary and "LLM summary unavailable" not in summary:
        for sig in ("```", "def ", "class ", "import ", "return "):
            if sig in summary:
                errors.append(f"LEAK: summary contains code signal: {sig!r}")
        print("  LLM summary generated — code leakage check: OK")
    else:
        print("  Note: LLM call failed — metadata-only fallback used.")

    for fp, content in codebase.items():
        slice_ = content[5:25].strip()
        if len(slice_) > 10 and slice_ in metadata_str:
            errors.append(f"LEAK: raw code slice from {fp!r} found in metadata")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("All assertions passed.")

    print()
    print("Metadata sample (first 3 files):")
    for fp, meta in list(result["metadata"].items())[:3]:
        funcs = meta.get("functions", [])[:8]
        classes = meta.get("classes", [])
        print(f"  {fp}:")
        print(f"    functions : {funcs}")
        if classes:
            print(f"    classes   : {classes}")


if __name__ == "__main__":
    asyncio.run(main())
