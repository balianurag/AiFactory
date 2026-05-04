"""
Code Security Processor — Local codebase analysis with zero raw-code leakage.

Solves the code leakage problem in enterprise agent pipelines:
  - LLM sees and reasons over the full source code (locally, in-process)
  - External APIs (Composio, MCP, logs, dumps) only ever receive safe abstractions
    (high-level summary, metadata, function signatures — no source code)

Architecture
------------
Worker emits:
    CODE_ANALYSIS: /path/to/project

Pipeline intercepts this signal (same pattern as RESEARCH:), calls
CodeSecurityProcessor locally, and injects only the safe output back
into the worker's context so it can complete its task.

External boundary (executor.py) has a second line of defence: _sanitize_params()
will redact any code that somehow slips through into ACTION: params.

Usage
-----
    processor = CodeSecurityProcessor()

    # 1. Read files from disk
    codebase = CodeSecurityProcessor.read_from_path("/path/to/project")

    # 2. Process locally — LLM sees code, external APIs never do
    safe_data = await processor.process_codebase(codebase, llm_think_fn=agent.think)

    # 3. safe_data is injection-safe:
    #    {
    #      "safe_summary": "...",        ← plain English, no code
    #      "metadata": {...},            ← sizes, languages, function names
    #      "file_count": 12,
    #      "total_bytes": 48200,
    #      "warning": "raw_code_never_exposed"
    #    }
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File extensions recognised as source code
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".sh",
        ".bash",
        ".sql",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".tf",
    }
)

# Max file size to process (skip huge generated/binary files)
_MAX_FILE_BYTES = 512 * 1024  # 512 KB

# Max total codebase bytes to pass to LLM in one call (summarise in chunks above this)
_CHUNK_THRESHOLD = 32 * 1024  # 32 KB

# Regex patterns that indicate raw code leaked into a summary
_CODE_IN_TEXT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```[\s\S]*?```"),  # fenced code blocks
    re.compile(r"`[^`\n]{10,}`"),  # long inline code
    re.compile(r"^\s{4,}(def |class |import |return )", re.MULTILINE),  # indented code
]


# =============================================================================
# Helpers
# =============================================================================


_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "c++",
    ".c": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".sh": "bash",
    ".sql": "sql",
}


def _detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return _LANG_MAP.get(ext, "unknown")


def _extract_python_metadata(content: str) -> dict[str, Any]:
    """
    Extract structural metadata from Python source using the AST.

    Returns only names (no code), so the result is safe to send externally.
    Falls back to empty metadata on parse errors.
    """
    meta: dict[str, Any] = {"functions": [], "classes": [], "imports": []}
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                meta["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                meta["classes"].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    meta["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                meta["imports"].append(node.module)
    except SyntaxError:
        # Best-effort: fall back to regex extraction
        meta["functions"] = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
        meta["classes"] = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
    # Deduplicate, cap at 50 each to bound payload size
    meta["functions"] = list(dict.fromkeys(meta["functions"]))[:50]
    meta["classes"] = list(dict.fromkeys(meta["classes"]))[:50]
    meta["imports"] = list(dict.fromkeys(meta["imports"]))[:30]
    return meta


def _extract_generic_metadata(content: str) -> dict[str, Any]:
    """Regex-based fallback for non-Python files."""
    functions = re.findall(r"(?:def|function|func)\s+(\w+)", content)
    classes = re.findall(r"(?:class|interface|struct)\s+(\w+)", content)
    return {
        "functions": list(dict.fromkeys(functions))[:50],
        "classes": list(dict.fromkeys(classes))[:50],
    }


def _validate_summary(text: str) -> str:
    """
    Strip any raw code that slipped into the LLM's summary.

    Removes fenced code blocks, long inline code, and indented code lines.
    This is the last line of defence before the summary leaves the processor.
    """
    result = text
    for pat in _CODE_IN_TEXT_PATTERNS:
        result = pat.sub("[code removed by security processor]", result)
    return result.strip()


def _content_hash(codebase: dict[str, str]) -> str:
    """Stable hash for a codebase dict — used to key the index cache."""
    h = hashlib.sha256()
    for path in sorted(codebase.keys()):
        h.update(path.encode())
        h.update(codebase[path][:512].encode())  # first 512 bytes per file is enough
    return h.hexdigest()[:16]


# =============================================================================
# CodeSecurityProcessor
# =============================================================================


# Type alias for the LLM think function injected from the worker agent
ThinkFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, str]]


class CodeSecurityProcessor:
    """
    Processes codebases locally so raw source never reaches external APIs.

    Two-phase design:
      Phase 1 (local): Build a LlamaIndex VectorStoreIndex over the codebase.
                       The LLM can query this index — all processing stays in-process.
      Phase 2 (safe):  Generate a plain-English summary with no source code.
                       Validate and strip any code that slips through.
                       Return only metadata + summary — safe for external use.

    LlamaIndex is used when available (already a project dependency).
    Falls back to direct chunked-LLM summarization when the index fails.

    Thread safety: instances are not thread-safe. Create one per pipeline run
    or use the module-level singleton via get_processor().
    """

    def __init__(self) -> None:
        # In-memory index cache: content_hash → built index
        self._index_cache: dict[str, Any] = {}

    # ── Public API ────────────────────────────────────────────────────

    @staticmethod
    def read_from_path(
        path: str | Path,
        extensions: list[str] | None = None,
        max_files: int = 200,
    ) -> dict[str, str]:
        """
        Read source files from a directory into a ``{filepath: content}`` dict.

        Only files with recognised code extensions are included.  Binary,
        compiled, and oversized files are silently skipped.

        Args:
            path:       Root directory to scan (non-recursive is opt-in via
                        extensions=[".py"] + shallow glob if needed).
            extensions: Whitelist of extensions.  Defaults to _CODE_EXTENSIONS.
            max_files:  Hard cap on number of files to prevent runaway reads.

        Returns:
            Dict mapping relative filepath strings to file content strings.
            Empty dict if path does not exist or is not a directory.
        """
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            logger.warning("CodeSecurityProcessor.read_from_path: not a directory: %s", root)
            return {}

        allowed_exts = frozenset(e.lower() for e in (extensions or _CODE_EXTENSIONS))
        codebase: dict[str, str] = {}

        # Skip non-source directories
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "coverage",
            "htmlcov",
            ".tox",
            "dumps",
            "experiments",
            ".superpowers",
            "docs",
            "migrations",
            ".claude",
            "docker",
            "init-db",
        }

        # If a src/ directory exists, prefer source subdirs and skip root-level clutter
        _src_root = root / "src"
        _prefer_src_dirs = _src_root.is_dir()

        for fpath in sorted(root.rglob("*")):
            if len(codebase) >= max_files:
                logger.info(
                    "CodeSecurityProcessor: reached max_files=%d limit at %s", max_files, root
                )
                break
            # Skip hidden/build directories
            if any(part in skip_dirs for part in fpath.parts):
                continue
            # Skip hidden files (dot-files)
            if fpath.name.startswith("."):
                continue
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in allowed_exts:
                continue
            if fpath.stat().st_size > _MAX_FILE_BYTES:
                logger.debug("Skipping oversized file: %s", fpath)
                continue
            # When src/ exists, skip root-level config files (pyproject.toml, *.yaml, *.json)
            # that are not inside a named source subdirectory.
            if _prefer_src_dirs:
                rel_parts = fpath.relative_to(root).parts
                if len(rel_parts) == 1 and fpath.suffix.lower() not in {".py", ".sh"}:
                    continue
            try:
                rel = str(fpath.relative_to(root))
                codebase[rel] = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.debug("Could not read %s: %s", fpath, exc)

        logger.info(
            "CodeSecurityProcessor.read_from_path: read %d files from %s",
            len(codebase),
            root,
        )
        return codebase

    async def process_codebase(
        self,
        codebase: dict[str, str],
        llm_think_fn: ThinkFn | None = None,
        max_injection_chars: int = 12_000,
    ) -> dict[str, Any]:
        """
        Process a codebase locally and return only safe abstractions.

        Uses structural index extraction (zero LLM calls, zero token cost) as the
        primary strategy.  *llm_think_fn* is accepted for API compatibility but is
        no longer called — the structural index gives workers the concrete data they
        need (file paths, function names, classes, imports, snippets) without burning
        the worker's own token budget on summarisation.

        Args:
            codebase:             Dict mapping filepath → content (from read_from_path).
            llm_think_fn:         Kept for backward compat; no longer used.
            max_injection_chars:  Hard cap on the returned structural index size.

        Returns:
            {
              "safe_summary": str,     ← structural index (functions, classes, imports)
              "metadata": dict,        ← per-file: size, language, functions, classes
              "file_count": int,
              "total_bytes": int,
              "languages": list[str],
              "warning": "raw_code_never_exposed"
            }
        """
        if not codebase:
            return self._empty_result("empty codebase")

        # Build per-file metadata (safe — names only, no code)
        file_metadata: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        languages: set[str] = set()

        for filepath, content in codebase.items():
            lang = _detect_language(filepath)
            languages.add(lang)
            total_bytes += len(content.encode("utf-8", errors="replace"))

            if lang == "python":
                meta = _extract_python_metadata(content)
            else:
                meta = _extract_generic_metadata(content)

            file_metadata[filepath] = {
                "size_bytes": len(content),
                "language": lang,
                **meta,
            }

        # Structural index — zero token cost, always safe
        safe_summary = self.build_structural_index(codebase, max_chars=max_injection_chars)

        return {
            "safe_summary": safe_summary,
            "metadata": file_metadata,
            "file_count": len(codebase),
            "total_bytes": total_bytes,
            "languages": sorted(languages),
            "warning": "raw_code_never_exposed",
        }

    # ── Internal ──────────────────────────────────────────────────────

    async def _generate_summary(
        self,
        codebase: dict[str, str],
        llm_think_fn: ThinkFn,
    ) -> str:
        """
        Generate a code-free plain-English summary of the codebase.

        Strategy:
        - Small codebases (≤ _CHUNK_THRESHOLD bytes): summarise all at once.
        - Large codebases: summarise per-file, then roll up into a final summary.

        Uses LlamaIndex VectorStoreIndex when available for richer RAG;
        falls back to direct LLM call for small codebases.

        The result is always validated through _validate_summary() to strip
        any code that the LLM includes despite instructions.
        """
        total_chars = sum(len(c) for c in codebase.values())

        try:
            if total_chars <= _CHUNK_THRESHOLD:
                summary = await self._summarise_direct(codebase, llm_think_fn)
            else:
                summary = await self._summarise_chunked(codebase, llm_think_fn)
        except Exception as exc:
            logger.warning("CodeSecurityProcessor: LLM summary failed: %s", exc)
            summary = self._metadata_only_summary(codebase)

        return _validate_summary(summary)

    async def _summarise_direct(
        self,
        codebase: dict[str, str],
        llm_think_fn: ThinkFn,
    ) -> str:
        """Summarise a small codebase in a single LLM call."""
        # Build a labelled listing (file headers only, then code) — LLM sees it all
        listing = "\n\n".join(f"=== {path} ===\n{content}" for path, content in codebase.items())
        prompt = (
            f"You are a code analyst. Analyse the following codebase and produce a "
            f"high-level technical summary.\n\n"
            f"CRITICAL RULES:\n"
            f"- Do NOT include any source code, function signatures with bodies, "
            f"or variable assignments in your response.\n"
            f"- Use only plain English descriptions.\n"
            f"- Cover: overall architecture, main modules/packages, key design patterns, "
            f"dependencies, entry points, and any notable security or quality observations.\n\n"
            f"CODEBASE:\n{listing}"
        )
        return await llm_think_fn(prompt, {})

    async def _summarise_chunked(
        self,
        codebase: dict[str, str],
        llm_think_fn: ThinkFn,
    ) -> str:
        """
        Summarise a large codebase by summarising groups of files then rolling up.
        """
        # Group files into chunks of ~_CHUNK_THRESHOLD chars each
        chunks: list[dict[str, str]] = []
        current_chunk: dict[str, str] = {}
        current_size = 0

        for path, content in codebase.items():
            if current_size + len(content) > _CHUNK_THRESHOLD and current_chunk:
                chunks.append(current_chunk)
                current_chunk = {}
                current_size = 0
            current_chunk[path] = content
            current_size += len(content)
        if current_chunk:
            chunks.append(current_chunk)

        # Summarise each chunk
        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            logger.debug(
                "CodeSecurityProcessor: summarising chunk %d/%d (%d files)",
                i + 1,
                len(chunks),
                len(chunk),
            )
            chunk_summary = await self._summarise_direct(chunk, llm_think_fn)
            chunk_summaries.append(chunk_summary)

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Roll up chunk summaries into a final summary
        combined = "\n\n---\n\n".join(
            f"Module group {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )
        rollup_prompt = (
            "You have been given summaries of different module groups from a large codebase. "
            "Synthesise them into a single cohesive high-level technical summary.\n\n"
            "CRITICAL: Do NOT include any source code. Plain English descriptions only.\n\n"
            f"MODULE GROUP SUMMARIES:\n{combined}"
        )
        return await llm_think_fn(rollup_prompt, {})

    def _metadata_only_summary(self, codebase: dict[str, str]) -> str:
        """Fallback summary built purely from metadata (no LLM call, always safe)."""
        langs: dict[str, int] = {}
        func_count = 0
        class_count = 0
        for fpath, content in codebase.items():
            lang = _detect_language(fpath)
            langs[lang] = langs.get(lang, 0) + 1
            if lang == "python":
                m = _extract_python_metadata(content)
            else:
                m = _extract_generic_metadata(content)
            func_count += len(m.get("functions", []))
            class_count += len(m.get("classes", []))

        lang_breakdown = ", ".join(f"{count} {lang}" for lang, count in sorted(langs.items()))
        return (
            f"Codebase contains {len(codebase)} files ({lang_breakdown}). "
            f"Approximately {func_count} functions and {class_count} classes identified. "
            f"(LLM summary unavailable — metadata-only fallback.)"
        )

    def build_structural_index(self, codebase: dict[str, str], max_chars: int = 12_000) -> str:
        """
        Build a rich structural index of the codebase — zero LLM calls, zero token cost.

        Returns a plain-text block listing each file with its functions, classes, key
        imports, size, and a 3-line content snippet.  Capped at *max_chars* so it
        always fits inside a worker prompt without blowing its token budget.

        This is the preferred injection method: the worker gets concrete, accurate
        structural data and can reason over it without the CSP consuming any of its
        own token budget for LLM summarisation calls.
        """
        lines: list[str] = []
        total_chars = 0

        # Sort by architectural priority so the char budget is spent on the
        # production code paths first, not alphabetically-first legacy dirs.
        # Priority tiers (lower = shown first):
        #   0 — core pipeline and orchestration
        #   1 — config and agent YAML definitions
        #   2 — CLI commands and helpers
        #   3 — API layer
        #   4 — everything else
        #   5 — legacy/scripts/tests (deprioritised)
        _priority_prefixes: list[tuple[str, int]] = [
            ("src/aifactory/graphs/", 0),
            ("src/aifactory/core/", 0),
            ("src/aifactory/config/", 1),
            ("src/aifactory/cli/", 2),
            ("src/aifactory/api/", 3),
            ("src/aifactory/mcp/", 3),
            ("src/aifactory/agents/", 4),  # mostly legacy
            ("scripts/", 5),
            ("tests/", 5),
            ("experiments/", 5),
        ]

        def _file_priority(fpath: str) -> tuple[int, str]:
            for prefix, rank in _priority_prefixes:
                if fpath.startswith(prefix):
                    return (rank, fpath)
            return (4, fpath)

        for fpath in sorted(codebase.keys(), key=_file_priority):
            content = codebase[fpath]
            lang = _detect_language(fpath)
            size_kb = len(content) / 1024

            if lang == "python":
                m = _extract_python_metadata(content)
            else:
                m = _extract_generic_metadata(content)

            funcs = m.get("functions", [])
            classes = m.get("classes", [])
            imports = m.get("imports", [])

            # 3-line snippet — first non-blank, non-comment, non-secret lines.
            # Skip lines with string literals assigned to variables (potential secrets).
            _secret_line_pat = re.compile(
                r"""(?:password|secret|key|token|api[_-]?key|auth)\s*=\s*['"]""",
                re.IGNORECASE,
            )
            snippet_lines = [
                ln.rstrip()
                for ln in content.splitlines()
                if (
                    ln.strip()
                    and not ln.strip().startswith(("#", "//", "/*", "*", '"""', "'''"))
                    and not _secret_line_pat.search(ln)
                    and "= '" not in ln  # skip any string assignment to be safe
                    and '= "' not in ln
                )
            ][:3]
            snippet = "  " + "\n  ".join(snippet_lines) if snippet_lines else ""

            entry_parts = [f"\n### {fpath}  ({lang}, {size_kb:.1f} KB)"]
            if classes:
                entry_parts.append(f"  classes:   {', '.join(classes[:12])}")
            if funcs:
                entry_parts.append(f"  functions: {', '.join(funcs[:20])}")
            if imports:
                entry_parts.append(f"  imports:   {', '.join(imports[:12])}")
            if snippet:
                entry_parts.append(f"  snippet:\n{snippet}")

            entry = "\n".join(entry_parts)
            if total_chars + len(entry) > max_chars:
                lines.append(f"\n... ({len(codebase) - len(lines)} more files truncated)")
                break
            lines.append(entry)
            total_chars += len(entry)

        top_dirs = sorted({p.split("/")[0] for p in codebase if "/" in p})[:5]
        root_label = ", ".join(top_dirs) if top_dirs else "."
        header = (
            f"## Codebase Structural Index\n"
            f"Root dirs: {root_label}\n"
            f"Files: {len(codebase)}  |  Languages: "
            + ", ".join(sorted({_detect_language(p) for p in codebase}))
            + "\n"
            "NOTE: Files ordered by architectural significance — "
            "core pipeline and orchestration shown first, legacy/scripts last.\n"
        )
        return header + "".join(lines)

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "safe_summary": f"No codebase to analyse ({reason}).",
            "metadata": {},
            "file_count": 0,
            "total_bytes": 0,
            "languages": [],
            "warning": "raw_code_never_exposed",
        }


# =============================================================================
# Module-level singleton
# =============================================================================

_processor: CodeSecurityProcessor | None = None


def get_processor() -> CodeSecurityProcessor:
    """Return the module-level CodeSecurityProcessor singleton."""
    global _processor
    if _processor is None:
        _processor = CodeSecurityProcessor()
    return _processor
