"""
GraphCompiler — translates an analyser-produced GraphSpec into a compiled LangGraph.

The four topology shapes:

  MINIMAL   master_receive → analyse → execution_lead → master_respond
            (3 active nodes; same as lite_mode but selected at plan-time by the analyser)

  STANDARD  master_receive → analyse → execution_lead → judge_review → master_respond
            (5 nodes; adds quality gate but no pre-analysis research loop)

  FULL      master_receive → analyse ⇄ researcher ⇄ reanalyse → execution_lead
              → judge_review → master_respond
            (7 nodes; current production default — full research loop + judge)

  EXTENDED  FULL + WaveGovernor integration placeholder
            (7 nodes now; WaveGovernor node added when Phase W3 lands)

The analyser emits a `graph_spec` JSON block in its output which `analyse` node
extracts and stores in shared_ctx["graph_spec"]. `build_dynamic_pipeline` delegates
to `GraphCompiler.compile()` when a valid spec is present; falls back to FULL topology
when the spec is absent or malformed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TopologyShape(StrEnum):
    """Topology complexity, ascending."""

    MINIMAL = "MINIMAL"
    STANDARD = "STANDARD"
    FULL = "FULL"
    EXTENDED = "EXTENDED"


@dataclass
class GraphSpec:
    """
    Structured description of the graph topology the analyser requests.

    Produced by the analyser node and consumed by GraphCompiler.compile().
    """

    topology: TopologyShape = TopologyShape.FULL
    needs_research: bool = False
    needs_judge: bool = True
    needs_integration: bool = False
    worker_strategy: str = "parallel"  # "parallel" | "sequential" | "single"
    task_count: int = 1
    reasoning: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphSpec:
        topo_raw = str(d.get("topology", "FULL")).upper()
        try:
            topology = TopologyShape(topo_raw)
        except ValueError:
            logger.warning("Unknown topology %r — defaulting to FULL", topo_raw)
            topology = TopologyShape.FULL
        return cls(
            topology=topology,
            needs_research=bool(d.get("needs_research", False)),
            needs_judge=bool(d.get("needs_judge", True)),
            needs_integration=bool(d.get("needs_integration", False)),
            worker_strategy=str(d.get("worker_strategy", "parallel")),
            task_count=int(d.get("task_count", 1)),
            reasoning=str(d.get("reasoning", "")),
        )

    @classmethod
    def from_json(cls, raw: str) -> GraphSpec:
        """Parse a JSON string (or JSON substring) into a GraphSpec."""
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract the first {...} block from a larger string
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    d = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.debug("GraphSpec JSON parse failed; using defaults")
                    return cls()
            else:
                return cls()
        return cls.from_dict(d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology": self.topology.value,
            "needs_research": self.needs_research,
            "needs_judge": self.needs_judge,
            "needs_integration": self.needs_integration,
            "worker_strategy": self.worker_strategy,
            "task_count": self.task_count,
            "reasoning": self.reasoning,
        }

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def is_lite(self) -> bool:
        """True when the topology collapses to single-model no-research mode."""
        return self.topology == TopologyShape.MINIMAL

    @property
    def has_research_loop(self) -> bool:
        return self.needs_research and self.topology in (
            TopologyShape.FULL,
            TopologyShape.EXTENDED,
        )

    @property
    def effective_topology(self) -> TopologyShape:
        """
        Resolve contradictions: e.g. STANDARD with needs_research=True → upgrade to FULL.
        """
        if self.needs_research and self.topology == TopologyShape.STANDARD:
            return TopologyShape.FULL
        if not self.needs_judge and self.topology == TopologyShape.FULL:
            return TopologyShape.STANDARD
        return self.topology


@dataclass
class CompileResult:
    """Return value from GraphCompiler.compile()."""

    graph: Any  # compiled LangGraph StateGraph
    topology: TopologyShape
    lite_mode: bool
    node_count: int
    spec: GraphSpec


class GraphCompiler:
    """
    Translates a GraphSpec into a compiled LangGraph graph.

    Usage::

        compiler = GraphCompiler(
            master=master_agent,
            analyser=analyser_agent,
            researcher=researcher_agent,
            lead=lead_agent,
            judge=judge_agent,
            **kwargs,
        )
        result = compiler.compile(spec)
        result.graph  # ready-to-run compiled LangGraph

    All node-factory imports are local so this module stays importable even when
    optional deps (qdrant, litellm) are absent.
    """

    def __init__(
        self,
        *,
        master: Any,
        analyser: Any,
        researcher: Any,
        lead: Any,
        judge: Any,
        semantic_cache: Any = None,
        memory_bridge: Any = None,
        research_cache: Any = None,
        embed_fn: Any = None,
        run_id: str = "",
        event_callback: Any = None,
        budget_tracker: Any = None,
        agent_cfgs: dict[str, Any] | None = None,
        models: list[Any] | None = None,
        mode: str = "balanced",
        registry: Any = None,
        code_processor: Any = None,
        checkpointer: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self._master = master
        self._analyser = analyser
        self._researcher = researcher
        self._lead = lead
        self._judge = judge
        self._semantic_cache = semantic_cache
        self._memory_bridge = memory_bridge
        self._research_cache = research_cache
        self._embed_fn = embed_fn
        self._run_id = run_id
        self._event_callback = event_callback
        self._budget_tracker = budget_tracker
        self._agent_cfgs = agent_cfgs or {}
        self._models = models or []
        self._mode = mode
        self._registry = registry
        self._code_processor = code_processor
        self._checkpointer = checkpointer
        self._knowledge_graph = knowledge_graph

    # ── Public API ───────────────────────────────────────────────────────────

    def compile(self, spec: GraphSpec | None = None) -> CompileResult:
        """
        Build and compile a LangGraph for the given spec.

        Falls back to FULL topology when spec is None.

        Week 5 KG integration: when a KnowledgeGraph is attached and detects
        integration/tool code, spec is upgraded to FULL (minimum) so the
        research loop and judge are always active for integration-heavy runs.
        """
        if spec is None:
            spec = GraphSpec()

        # W5: KG → graph_spec: boost topology when integration code is present
        spec = self._kg_upgrade_spec(spec)

        topology = spec.effective_topology

        if topology == TopologyShape.MINIMAL:
            return self._compile_minimal(spec)
        elif topology == TopologyShape.STANDARD:
            return self._compile_standard(spec)
        elif topology == TopologyShape.EXTENDED:
            return self._compile_full(spec, extended=True)
        else:
            return self._compile_full(spec)

    def _kg_upgrade_spec(self, spec: GraphSpec) -> GraphSpec:
        """
        Use KnowledgeGraph signals to upgrade spec topology when needed.

        Rules:
          - KG detects integration files + spec is MINIMAL/STANDARD → upgrade to FULL
          - KG has many entities (complex codebase) + spec is MINIMAL → upgrade to STANDARD
        """
        if self._knowledge_graph is None:
            return spec
        try:
            has_integration = self._knowledge_graph.has_integration_code()
            kg_stats = self._knowledge_graph.stats()
            entity_count = kg_stats.get("total_entities", 0)

            if has_integration and spec.topology in (TopologyShape.MINIMAL, TopologyShape.STANDARD):
                logger.info(
                    "KG upgrade: integration code detected → upgrading %s to FULL",
                    spec.topology.value,
                )
                from dataclasses import replace

                return replace(spec, topology=TopologyShape.FULL, needs_research=True)

            if entity_count > 200 and spec.topology == TopologyShape.MINIMAL:
                logger.info(
                    "KG upgrade: large codebase (%d entities) → upgrading MINIMAL to STANDARD",
                    entity_count,
                )
                from dataclasses import replace

                return replace(spec, topology=TopologyShape.STANDARD)
        except Exception as exc:
            logger.debug("KG spec upgrade skipped: %s", exc)
        return spec

    # ── Topology builders ────────────────────────────────────────────────────

    def _compile_minimal(self, spec: GraphSpec) -> CompileResult:
        """
        MINIMAL: master_receive → analyse → execution_lead → master_respond
        No research, no judge LLM call. Equivalent to lite_mode=True.
        """
        from langgraph.graph import END, START, StateGraph

        from aifactory.graphs.dynamic_pipeline import (
            DynamicPipelineState,
            make_analyse,
            make_master_receive,
            make_master_respond,
            route_after_analysis,
        )

        g = StateGraph(DynamicPipelineState)
        g.add_node(
            "master_receive",
            make_master_receive(self._master, semantic_cache=self._semantic_cache),
        )
        g.add_node("analyse", make_analyse(self._analyser))
        g.add_node("execution_lead", self._make_execution_lead_node(lite_mode=True))
        g.add_node(
            "master_respond",
            make_master_respond(
                self._master, semantic_cache=self._semantic_cache, run_id=self._run_id
            ),
        )

        g.add_edge(START, "master_receive")
        g.add_edge("master_receive", "analyse")
        g.add_conditional_edges(
            "analyse",
            route_after_analysis,
            {
                "master_respond": "master_respond",
                "researcher": "execution_lead",  # no researcher in MINIMAL
                "execution_lead": "execution_lead",
            },
        )
        g.add_edge("execution_lead", "master_respond")
        g.add_edge("master_respond", END)

        compiled = g.compile(checkpointer=self._checkpointer)
        logger.info("GraphCompiler: compiled MINIMAL topology (4 nodes, lite_mode=True)")
        return CompileResult(
            graph=compiled,
            topology=TopologyShape.MINIMAL,
            lite_mode=True,
            node_count=4,
            spec=spec,
        )

    def _compile_standard(self, spec: GraphSpec) -> CompileResult:
        """
        STANDARD: master_receive → analyse → execution_lead → judge_review → master_respond
        No research loop. Adds quality gate.
        """
        from langgraph.graph import END, START, StateGraph

        from aifactory.graphs.dynamic_pipeline import (
            DynamicPipelineState,
            make_analyse,
            make_judge_review,
            make_master_receive,
            make_master_respond,
            route_after_analysis,
        )

        g = StateGraph(DynamicPipelineState)
        g.add_node(
            "master_receive",
            make_master_receive(self._master, semantic_cache=self._semantic_cache),
        )
        g.add_node("analyse", make_analyse(self._analyser))
        g.add_node("execution_lead", self._make_execution_lead_node(lite_mode=False))
        g.add_node(
            "judge_review",
            make_judge_review(
                self._judge,
                self._registry,
                self._run_id,
                lite_mode=False,
                event_callback=self._event_callback,
            ),
        )
        g.add_node(
            "master_respond",
            make_master_respond(
                self._master, semantic_cache=self._semantic_cache, run_id=self._run_id
            ),
        )

        g.add_edge(START, "master_receive")
        g.add_edge("master_receive", "analyse")
        g.add_conditional_edges(
            "analyse",
            route_after_analysis,
            {
                "master_respond": "master_respond",
                "researcher": "execution_lead",  # no researcher in STANDARD
                "execution_lead": "execution_lead",
            },
        )
        g.add_edge("execution_lead", "judge_review")
        g.add_edge("judge_review", "master_respond")
        g.add_edge("master_respond", END)

        compiled = g.compile(checkpointer=self._checkpointer)
        logger.info("GraphCompiler: compiled STANDARD topology (5 nodes)")
        return CompileResult(
            graph=compiled,
            topology=TopologyShape.STANDARD,
            lite_mode=False,
            node_count=5,
            spec=spec,
        )

    def _compile_full(self, spec: GraphSpec, extended: bool = False) -> CompileResult:
        """
        FULL:     master_receive → analyse ⇄ researcher ⇄ reanalyse → execution_lead
                    → judge_review → master_respond  (7 nodes)

        EXTENDED: same as FULL + wave_governor node between execution_lead and
                    judge_review, with a conditional edge that can loop back to
                    execution_lead for mode-escalated re-runs.  (8 nodes)
        """
        from langgraph.graph import END, START, StateGraph

        from aifactory.graphs.dynamic_pipeline import (
            DynamicPipelineState,
            make_analyse,
            make_judge_review,
            make_master_receive,
            make_master_respond,
            make_reanalyse,
            make_researcher,
            route_after_analysis,
            route_after_reanalyse,
            route_after_researcher,
        )

        g = StateGraph(DynamicPipelineState)
        g.add_node(
            "master_receive",
            make_master_receive(self._master, semantic_cache=self._semantic_cache),
        )
        g.add_node("analyse", make_analyse(self._analyser))
        g.add_node(
            "researcher",
            make_researcher(
                self._researcher,
                bridge=self._memory_bridge,
                research_cache=self._research_cache,
                run_id=self._run_id,
                embed_fn=self._embed_fn,
            ),
        )
        g.add_node("reanalyse", make_reanalyse(self._analyser))
        g.add_node("execution_lead", self._make_execution_lead_node(lite_mode=False))
        g.add_node(
            "judge_review",
            make_judge_review(
                self._judge,
                self._registry,
                self._run_id,
                lite_mode=False,
                event_callback=self._event_callback,
            ),
        )
        g.add_node(
            "master_respond",
            make_master_respond(
                self._master, semantic_cache=self._semantic_cache, run_id=self._run_id
            ),
        )

        g.add_edge(START, "master_receive")
        g.add_edge("master_receive", "analyse")
        g.add_conditional_edges(
            "analyse",
            route_after_analysis,
            {
                "master_respond": "master_respond",
                "researcher": "researcher",
                "execution_lead": "execution_lead",
            },
        )
        g.add_conditional_edges(
            "researcher",
            route_after_researcher,
            {
                "reanalyse": "reanalyse",
                "execution_lead": "execution_lead",
            },
        )
        g.add_conditional_edges(
            "reanalyse",
            route_after_reanalyse,
            {
                "researcher": "researcher",
                "execution_lead": "execution_lead",
            },
        )

        topology = TopologyShape.EXTENDED if extended else TopologyShape.FULL
        node_count = 7

        if extended:
            # EXTENDED: add WaveGovernor between execution_lead and judge_review
            from aifactory.graphs.wave_governor import (
                WaveGovernor,
                make_wave_governor,
                route_after_wave_governor,
            )

            _governor = WaveGovernor(event_callback=self._event_callback)
            g.add_node(
                "wave_governor",
                make_wave_governor(_governor, current_mode=self._mode),
            )
            g.add_edge("execution_lead", "wave_governor")
            g.add_conditional_edges(
                "wave_governor",
                route_after_wave_governor,
                {
                    "execution_lead": "execution_lead",
                    "judge_review": "judge_review",
                },
            )
            node_count = 8
            logger.info("GraphCompiler: EXTENDED topology with WaveGovernor activated")
        else:
            g.add_edge("execution_lead", "judge_review")

        g.add_edge("judge_review", "master_respond")
        g.add_edge("master_respond", END)

        compiled = g.compile(checkpointer=self._checkpointer)
        logger.info(
            "GraphCompiler: compiled %s topology (%d nodes, 3 conditional edges)",
            topology.value,
            node_count,
        )
        return CompileResult(
            graph=compiled, topology=topology, lite_mode=False, node_count=node_count, spec=spec
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_execution_lead_node(self, *, lite_mode: bool) -> Any:
        from aifactory.graphs.dynamic_pipeline import make_execution_lead

        return make_execution_lead(
            self._lead,
            self._agent_cfgs.get("worker", {}),
            self._models,
            self._mode,
            self._registry,
            bridge=self._memory_bridge,
            event_callback=self._event_callback,
            budget_tracker=self._budget_tracker,
            researcher=self._researcher,
            code_processor=self._code_processor,
        )

    # ── Class-level factory (convenience) ───────────────────────────────────

    @staticmethod
    def spec_from_shared_ctx(shared_ctx: dict[str, Any]) -> GraphSpec | None:
        """
        Extract a GraphSpec from the shared context dict produced by the analyse node.

        Returns None when no graph_spec key is present.
        """
        raw = shared_ctx.get("graph_spec")
        if not raw:
            return None
        if isinstance(raw, dict):
            return GraphSpec.from_dict(raw)
        if isinstance(raw, str):
            return GraphSpec.from_json(raw)
        return None
