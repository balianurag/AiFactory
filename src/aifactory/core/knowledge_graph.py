"""
KnowledgeGraph — lightweight in-process entity/relation store for the Adaptive Mesh Pipeline.

Design goals:
  - Zero external deps beyond stdlib (no Neo4j, no graph DB required).
  - Fed by CodeSecurityProcessor.build_structural_index() at run start.
  - diff_since(snapshot_id) enables incremental re-scans so CSP only re-processes
    changed files (tracked by content hash).
  - Queried by memory_router for context injection and by GraphCompiler for
    topology hints (e.g. "does this run touch integration code?").

Entity types:  file, function, class_, import_, module
Relation types: contains, imports, calls, inherits, depends_on

Storage: two dicts — _entities: {id → EntityNode}, _relations: {id → RelationEdge}
Snapshots: {snapshot_id → {file_path → content_hash}} so diff_since() returns only
the files whose hash changed (or were added/removed) since that snapshot.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class EntityNode:
    id: str
    kind: str  # file | function | class_ | import_ | module
    name: str
    file_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationEdge:
    id: str
    source_id: str
    target_id: str
    relation: str  # contains | imports | calls | inherits | depends_on
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGSnapshot:
    snapshot_id: str
    timestamp: float
    file_hashes: dict[str, str]  # file_path → sha256[:16]


@dataclass
class KGDiff:
    snapshot_id: str
    added: list[str]
    modified: list[str]
    removed: list[str]

    @property
    def changed(self) -> list[str]:
        return self.added + self.modified

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.removed)


# ── Main class ────────────────────────────────────────────────────────────────


class KnowledgeGraph:
    """
    In-process entity/relation store populated from CSP structural index output.

    Typical usage::

        kg = KnowledgeGraph()
        snap_id = kg.ingest_structural_index(csp.build_structural_index(codebase))
        # ... later run ...
        diff = kg.diff_since(snap_id)
        if not diff.is_empty:
            kg.ingest_structural_index(csp.build_structural_index(codebase), snap_id)

    Or directly from a raw codebase metadata dict::

        snap_id = kg.ingest_codebase(codebase_metadata)
    """

    def __init__(self) -> None:
        self._entities: dict[str, EntityNode] = {}
        self._relations: dict[str, RelationEdge] = {}
        self._snapshots: dict[str, KGSnapshot] = {}
        self._file_entity_index: dict[str, list[str]] = {}  # file_path → [entity_ids]

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_codebase(
        self,
        codebase_meta: dict[str, dict[str, Any]],
        base_snapshot_id: str | None = None,
    ) -> str:
        """
        Ingest structured metadata (output of _extract_python_metadata per file).

        codebase_meta: {file_path → {"functions": [...], "classes": [...], "imports": [...]}}

        Returns a snapshot_id identifying this ingestion point (use with diff_since).
        """
        snapshot_id = _make_id("snap", str(time.time()))
        file_hashes: dict[str, str] = {}

        for fpath, meta in codebase_meta.items():
            content_repr = str(sorted(meta.items()))
            fhash = _sha16(content_repr)
            file_hashes[fpath] = fhash

            # Only re-index if file changed since last snapshot
            if base_snapshot_id:
                prev = self._snapshots.get(base_snapshot_id)
                if prev and prev.file_hashes.get(fpath) == fhash:
                    continue

            self._index_file(fpath, meta)

        # Handle removals: files present in base but absent now
        if base_snapshot_id:
            prev = self._snapshots.get(base_snapshot_id)
            if prev:
                for old_fpath in prev.file_hashes:
                    if old_fpath not in file_hashes:
                        self._remove_file_entities(old_fpath)

        self._snapshots[snapshot_id] = KGSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            file_hashes=file_hashes,
        )
        return snapshot_id

    def ingest_structural_index(
        self,
        index_text: str,
        base_snapshot_id: str | None = None,
    ) -> str:
        """
        Parse the plain-text structural index produced by CSP.build_structural_index()
        and ingest it into the graph.

        Format expected::

            ### src/foo.py  (python, 1.2 KB)
              classes:   Foo, Bar
              functions: do_thing, helper
              imports:   os, json, typing
        """
        meta = _parse_structural_index(index_text)
        return self.ingest_codebase(meta, base_snapshot_id=base_snapshot_id)

    # ── Diff / incremental re-scan ────────────────────────────────────────────

    def diff_since(self, snapshot_id: str, current_hashes: dict[str, str] | None = None) -> KGDiff:
        """
        Return which files changed since snapshot_id.

        If current_hashes is provided (file_path → sha256[:16]) it is compared
        directly. Otherwise compares the two most recent snapshots.
        """
        base = self._snapshots.get(snapshot_id)
        if base is None:
            return KGDiff(snapshot_id=snapshot_id, added=[], modified=[], removed=[])

        if current_hashes is None:
            # Compare base against the latest snapshot (if any)
            latest = max(self._snapshots.values(), key=lambda s: s.timestamp, default=None)
            if latest is None or latest.snapshot_id == snapshot_id:
                return KGDiff(snapshot_id=snapshot_id, added=[], modified=[], removed=[])
            current_hashes = latest.file_hashes

        added = [f for f in current_hashes if f not in base.file_hashes]
        removed = [f for f in base.file_hashes if f not in current_hashes]
        modified = [
            f
            for f in current_hashes
            if f in base.file_hashes and current_hashes[f] != base.file_hashes[f]
        ]
        return KGDiff(
            snapshot_id=snapshot_id,
            added=added,
            modified=modified,
            removed=removed,
        )

    def hash_codebase(self, codebase: dict[str, str]) -> dict[str, str]:
        """Return {file_path → sha16} for a raw codebase dict (content values)."""
        return {fp: _sha16(content) for fp, content in codebase.items()}

    # ── Query API ─────────────────────────────────────────────────────────────

    def entities_for_file(self, file_path: str) -> list[EntityNode]:
        ids = self._file_entity_index.get(file_path, [])
        return [self._entities[i] for i in ids if i in self._entities]

    def find_entities(
        self,
        kind: str | None = None,
        name_contains: str | None = None,
        file_path: str | None = None,
    ) -> list[EntityNode]:
        results = []
        for e in self._entities.values():
            if kind and e.kind != kind:
                continue
            if name_contains and name_contains.lower() not in e.name.lower():
                continue
            if file_path and e.file_path != file_path:
                continue
            results.append(e)
        return results

    def relations_for(self, entity_id: str) -> list[RelationEdge]:
        return [r for r in self._relations.values() if r.source_id == entity_id]

    def integration_files(self) -> list[str]:
        """Return file paths that appear to be integration/tool code."""
        signals = ("mcp", "composio", "integration", "tool", "provider", "connector")
        return [
            e.file_path
            for e in self._entities.values()
            if e.kind == "file" and any(sig in e.file_path.lower() for sig in signals)
        ]

    def has_integration_code(self) -> bool:
        return bool(self.integration_files())

    def module_names(self) -> list[str]:
        return [e.name for e in self._entities.values() if e.kind == "module"]

    def function_names(self, file_path: str | None = None) -> list[str]:
        entities = self.find_entities(kind="function", file_path=file_path)
        return [e.name for e in entities]

    def class_names(self, file_path: str | None = None) -> list[str]:
        entities = self.find_entities(kind="class_", file_path=file_path)
        return [e.name for e in entities]

    def stats(self) -> dict[str, int]:
        kind_counts: dict[str, int] = {}
        for e in self._entities.values():
            kind_counts[e.kind] = kind_counts.get(e.kind, 0) + 1
        return {
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "snapshots": len(self._snapshots),
            **kind_counts,
        }

    def to_summary(self, max_entries: int = 30) -> str:
        """Return a compact plain-text KG summary for injection into agent prompts."""
        lines = [f"## Knowledge Graph ({len(self._entities)} entities)"]
        files = self.find_entities(kind="file")[:max_entries]
        for f in files:
            funcs = self.function_names(f.name)
            classes = self.class_names(f.name)
            parts = [f"  {f.name}"]
            if classes:
                parts.append(f"classes={','.join(classes[:5])}")
            if funcs:
                parts.append(f"fns={','.join(funcs[:8])}")
            lines.append("  ".join(parts))
        if len(self._entities) > max_entries:
            lines.append(f"  ... ({len(self._entities) - max_entries} more entities)")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _index_file(self, fpath: str, meta: dict[str, Any]) -> None:
        # Remove existing entities for this file before re-indexing
        self._remove_file_entities(fpath)

        entity_ids: list[str] = []

        # File entity
        file_id = _make_id("file", fpath)
        self._entities[file_id] = EntityNode(id=file_id, kind="file", name=fpath, file_path=fpath)
        entity_ids.append(file_id)

        # Function entities
        for fname in meta.get("functions", []):
            fid = _make_id("fn", f"{fpath}:{fname}")
            self._entities[fid] = EntityNode(id=fid, kind="function", name=fname, file_path=fpath)
            entity_ids.append(fid)
            # file contains function
            rid = _make_id("rel", f"{file_id}:contains:{fid}")
            self._relations[rid] = RelationEdge(
                id=rid, source_id=file_id, target_id=fid, relation="contains"
            )

        # Class entities
        for cname in meta.get("classes", []):
            cid = _make_id("cls", f"{fpath}:{cname}")
            self._entities[cid] = EntityNode(id=cid, kind="class_", name=cname, file_path=fpath)
            entity_ids.append(cid)
            rid = _make_id("rel", f"{file_id}:contains:{cid}")
            self._relations[rid] = RelationEdge(
                id=rid, source_id=file_id, target_id=cid, relation="contains"
            )

        # Import entities
        for imp in meta.get("imports", []):
            imp_id = _make_id("imp", f"{fpath}:{imp}")
            self._entities[imp_id] = EntityNode(
                id=imp_id, kind="import_", name=imp, file_path=fpath
            )
            entity_ids.append(imp_id)
            rid = _make_id("rel", f"{file_id}:imports:{imp_id}")
            self._relations[rid] = RelationEdge(
                id=rid, source_id=file_id, target_id=imp_id, relation="imports"
            )

        self._file_entity_index[fpath] = entity_ids

    def _remove_file_entities(self, fpath: str) -> None:
        ids_to_remove = self._file_entity_index.pop(fpath, [])
        for eid in ids_to_remove:
            self._entities.pop(eid, None)
        # Remove relations that reference removed entities
        to_del = [
            rid
            for rid, r in self._relations.items()
            if r.source_id in ids_to_remove or r.target_id in ids_to_remove
        ]
        for rid in to_del:
            self._relations.pop(rid, None)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _make_id(prefix: str, key: str) -> str:
    return f"{prefix}:{_sha16(key)}"


def _parse_structural_index(text: str) -> dict[str, dict[str, Any]]:
    """
    Parse CSP structural index plain text into {file_path → meta_dict}.

    Handles the format::

        ### src/foo.py  (python, 1.2 KB)
          classes:   Foo, Bar
          functions: do_thing, helper
          imports:   os, json
    """
    result: dict[str, dict[str, Any]] = {}
    current_file: str | None = None
    current_meta: dict[str, Any] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            if current_file:
                result[current_file] = current_meta
            # Parse "### src/foo.py  (python, 1.2 KB)"
            header = line[4:].split("  ")[0].strip()
            current_file = header
            current_meta = {}
        elif current_file:
            if line.startswith("classes:"):
                val = line.split(":", 1)[1].strip()
                current_meta["classes"] = [c.strip() for c in val.split(",") if c.strip()]
            elif line.startswith("functions:"):
                val = line.split(":", 1)[1].strip()
                current_meta["functions"] = [f.strip() for f in val.split(",") if f.strip()]
            elif line.startswith("imports:"):
                val = line.split(":", 1)[1].strip()
                current_meta["imports"] = [i.strip() for i in val.split(",") if i.strip()]

    if current_file:
        result[current_file] = current_meta

    return result
