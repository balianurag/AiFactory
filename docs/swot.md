# AiFactory — SWOT & Path to a Sellable SaaS

_Last updated: 2026-07-18. Grounded in the actual codebase, not a generic template._

## Verdict in one line
**Strong engine, unproven trust, no wedge.** The tech is ahead of where most solo SaaS starts; the gaps are quality-assurance and positioning — both fixable, both matter more than any new feature.

---

## 🟢 Strengths (verified in code)
- **Cost-efficiency engine is a genuine moat** — ComplexityGate (bypass → fast_llm → full), semantic cache, prompt caching, RL-lite model selection, per-run USD budget caps. Demonstrable "same output, fraction of the cost" story vs token-hungry CrewAI/AutoGen.
- **Provider-agnostic** — LiteLLM proxy → 100+ models, no vendor lock-in, per-task price/quality arbitrage.
- **Config-driven agents (YAML)** — new roles need zero code; future marketplace angle.
- **Production hardening most early products lack** — 1505 tests, circuit breakers, rate limiting, gVisor/seccomp sandbox, LangSmith/OTel observability, cross-run memory.
- **Enterprise-security primitives built** — CodeSecurityProcessor (strips local code pre-LLM), MCP catalog + deploy-mode routing for data residency.
- **Full stack exists** — API + WebSocket, Tauri desktop app, web UI, Supabase auth/RBAC/multitenancy. Solo-founder buildable.

## 🔴 Weaknesses (honest)
- **Output-quality reliability unproven** — 4 defects just found/fixed (findings dropped, memory pollution, judge metrics silently missing, revenue figures rescaled). **No systematic eval harness.** #1 gap for a paid product.
- **Complexity has outrun coherence** — 40+ design decisions, W1–W6 adaptive mesh, gate-score/lite-mode contradiction (routes to "full" but silently drops the judge). Over-engineering risk.
- **No ICP or positioning** — horizontal "any idea → output" engine. Market size ≠ distribution.
- **Latency & COGS** — multi-agent runs 2–3 min; tokens are the COGS. UX and margin risk under load.
- **Self-host fragility & bus factor** — Zscaler/cert coupling, DB fallbacks, no working CI, single maintainer.

## 🔵 Opportunities
- **2026 agentic-AI tailwind** — "digital employees" for SMBs.
- **Vertical wedge = fastest revenue** — pick one (CA firms / GST reconciliation, or Indic content repurposing). Lead with cost-efficiency.
- **Hybrid GTM** — done-for-you agent builds ($1.5–5k) now while self-serve matures; doubles as market research.
- **Data-residency / on-prem** for regulated buyers — a niche model vendors won't prioritize.
- **Agent-config marketplace** — ecosystem around the YAML format.

## ⚠️ Threats
- **Platform risk** — built on LangGraph; competes with CrewAI, AutoGen, Dify, Flowise, Lindy, n8n; OpenAI/Anthropic native orchestration could commoditize the layer.
- **Model vendors moving up the stack.**
- **Trust/liability** — wrong output in finance/legal = liability. Needs eval + human-in-loop.
- **Fast churn** — standards (e.g., MCP) could obsolete "self-modifying graph."

---

## TOWS — strategic moves
| | Leverage Strengths | Fix Weaknesses |
|---|---|---|
| **Seize Opportunities** | **SO:** GTM on cost-efficiency + one vertical. | **WO:** Build eval harness before selling into finance/legal — turn weakness into trust feature. |
| **Counter Threats** | **ST:** Defend platform risk with on-prem/data-residency + self-improving moat. | **WT:** Don't compete horizontally with LangGraph/CrewAI. Narrow to a winnable vertical. |

## Sellable-SaaS readiness scorecard
| Dimension | Status | Gap |
|---|---|---|
| Core tech | 🟢 Strong | — |
| Cost engineering | 🟢 Strong (differentiator) | Make it a visible buyer metric |
| Output quality/trust | 🔴 Weak | **Eval harness + quality SLA — blocker** |
| Positioning / ICP | 🔴 None | Pick one vertical |
| Multitenancy/auth | 🟡 Scaffolded | Harden, load-test |
| Infra / ops | 🟡 Dev-grade | Cloud-native, CI, de-Zscaler |
| Pricing model | 🔴 Undefined | Usage-based w/ margin guardrails |

## Sequenced roadmap (highest leverage first)
1. **Positioning + ICP** — pick ONE vertical, define ICP + value prop. Unblocks everything.
2. **Quality eval harness** — scoped to that vertical. The trust blocker.
3. **Fix gate-score/lite-mode contradiction** — coherence before features.
4. **Package cost-efficiency as a demo metric** — "$0.02 vs $0.15 on raw GPT-5."

## Skill → task map (from alirezarezvani/claude-skills)
| Roadmap item | Relevant skill |
|---|---|
| Positioning / ICP | `strategist`, `product-manager` |
| Vertical validation | `ux-researcher`, `experiment-designer` |
| Eval harness | `api-test-suite-builder`, `pr-review-expert` |
| Pricing / metrics | `saas-metrics-coach` |
| Launch / messaging | `growth-marketer`, `launch-strategy`, `copywriting` |
| CI / ops hardening | `ci-cd-pipeline-builder`, `dependency-auditor` |
