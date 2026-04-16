# MCP Catalog Feasibility Report
## Replacing / Complementing Composio as the Tool Integration Layer in AiFactory

**Date:** April 2026  
**Author:** Research via Claude Code  
**Status:** Decision-pending

---

## Executive Summary

- **MCP ecosystem is production-ready for 5–6 of the 10 target services**, but coverage is uneven: GitHub (28.8k-star official server) and Atlassian/Jira (4.9k-star, 72 tools) are excellent, while Microsoft Teams, Linear, and Gmail have only community implementations of varying maturity.
- **A full Composio replacement is premature today.** The per-tenant OAuth infrastructure, token vault, and N-server multiplexer that would be needed are non-trivial to build and maintain; rebuilding them from scratch is 4–8 weeks of infra work before any business-level feature is added.
- **A hybrid approach is the pragmatic path**: replace Composio with direct MCP servers for the 4–5 highest-demand services (GitHub, Jira/Confluence, Notion, Google Workspace) where the official servers are mature. Keep Composio as fallback for the long tail (Linear, Teams, Gmail, 900+ others).
- **Long-term (12+ months), a full MCP catalog is the correct architecture** for a B2B enterprise product: direct API calls eliminate the Composio data-routing concern, enable on-premise deployment, and give you full control over auth, rate limits, and tool coverage.

---

## 1. MCP Ecosystem Status

### 1.1 Protocol Maturity

The Model Context Protocol is maintained by Anthropic and has reached protocol version **2025-06-18** (Streamable HTTP transport replaces the earlier SSE-only transport). The spec is stable: JSON-RPC 2.0 over two standard transports (stdio subprocess and Streamable HTTP with optional SSE). The Python SDK (`mcp` package via `mcp-python-sdk`) implements the full spec and is production-quality. OAuth 2.1 authorization is formally specified for HTTP transport; stdio servers retrieve credentials from environment variables.

### 1.2 Available MCP Servers for Target Services

| Service | Best Available Server | Maintainer | Stars | Last Active | Tools / Notes |
|---|---|---|---|---|---|
| **GitHub** | `github/github-mcp-server` | Official (GitHub) | 28,800 | Jan 2025 | Repos, issues, PRs, Actions, code security. Toolsets configurable. |
| **Jira + Confluence** | `sooperset/mcp-atlassian` | Community (high quality) | 4,900 | Apr 2026 | 72 tools, Cloud + Server/DataCenter, CQL/JQL search |
| **Notion** | `makenotion/notion-mcp-server` | Official (Notion) | 4,200 | ~2026 | 22 tools, query/create/update pages + databases, token-optimized |
| **Google Workspace / Gmail** | `taylorwilsdon/google_workspace_mcp` | Community | 2,100 | Apr 2026 | Gmail, Drive, Docs, Sheets, Calendar via OAuth. No official Google server. |
| **Slack** | `korotovsky/slack-mcp-server` | Community | 1,500 | Apr 2026 | Full messaging, DMs, history, App support, GovSlack. Written in Go. |
| **GitLab** | Archived reference impl | Archived (community) | N/A | Stale | Original MCP-servers repo entry; moved to third-party maintainers |
| **Microsoft Teams** | `InditexTech/mcp-teams-server` | Community | 365 | Apr 2026 | Python, messaging. Microsoft official server has 4 stars (C#, demo only). |
| **Linear** | `jerhadf/linear-mcp-server` | Community | 345 | May 2025 | JS, basic issue/project management. No official Linear server. |
| **Linear (alt)** | `tacticlaunch/mcp-linear` | Community | 132 | Sep 2025 | TypeScript, more complete |
| **Atlassian (official)** | `atlassian/mcp-atlassian` | Official (Atlassian) | Listed in MCP registry | Apr 2026 | Official; listed in modelcontextprotocol.io registry |

**Gaps:**
- **Microsoft Teams**: No credible official server. The top community option (365 stars) is adequate for messaging but Graph API breadth varies.
- **Linear**: Community-only, partially stale. Composio's Linear support is more battle-tested.
- **Gmail / Google Workspace**: Community-only; Google has published no official server. `taylorwilsdon/google_workspace_mcp` is well-maintained but carries community support risk.
- **GitLab**: Reference implementation was archived; community forks exist but fragmented.

---

## 2. MCP Technical Architecture

### 2.1 Transport Mechanisms

| Transport | How It Works | Best For | Auth |
|---|---|---|---|
| **stdio** | MCP client spawns server as subprocess; JSON-RPC over stdin/stdout | Local-only; one connection per process | Env vars / API keys at launch |
| **Streamable HTTP** | Server runs as long-lived HTTP process; POST for requests, optional SSE stream for responses | Remote / multi-tenant / containerized | OAuth 2.1 Bearer tokens, API keys in headers |

For AiFactory's architecture (async Python pipeline, Docker-first deployment), **Streamable HTTP is the appropriate transport** for enterprise deployments: servers run as sidecar containers or separate services, and the pipeline holds long-lived HTTP sessions per tenant.

For local dev or single-tenant internal use, stdio is simpler (no auth surface, zero infrastructure), but doesn't scale to multi-tenant B2B.

### 2.2 How Agents Would Call MCP Servers

Current AiFactory flow:
```
Worker LLM output → ACTION:composio\n{...}\nACTION_END → executor.py → ComposioProvider → Composio API → result injected back
```

MCP catalog flow:
```
Worker LLM output → MCP tool call (tools/call JSON-RPC) → MCPClientPool → selected server → upstream API → result
```

The key change is that instead of dispatching ACTION: text blocks through `executor.py`, the pipeline would:

1. At pipeline start, enumerate connected MCP servers (via `tools/list` from each server).
2. Inject the tool manifest into worker system prompts (replacing `get_tools_summary_for_prompt()`).
3. After worker LLM output, detect structured tool-call format (either OpenAI `tool_calls` format or a lightweight replacement for the current ACTION: block).
4. Route to the appropriate MCP server's `tools/call` endpoint.
5. Inject result back into the worker's context.

The `IntegrationRegistry` + `BaseIntegration` abstraction already exists and would be extended: a new `MCPServerIntegration(BaseIntegration)` class wraps an `mcp.ClientSession` and delegates `execute_action()` to `session.call_tool()`.

### 2.3 Multiplexing N Servers in Python Async

The Python MCP SDK (`mcp`) provides `ClientSession` with async context managers. Each server gets one `ClientSession`:

```python
# Conceptual MCPClientPool
class MCPClientPool:
    sessions: dict[str, ClientSession]  # server_name → session

    async def call_tool(self, server: str, tool: str, args: dict) -> CallToolResult:
        return await self.sessions[server].call_tool(tool, arguments=args)
```

Since AiFactory already uses asyncio throughout and LangGraph async nodes, this maps cleanly. The pool initialises once per pipeline run (or once per deployment for HTTP-transport servers) and routes calls by server name. Workers do not need process isolation — they share the pool via the existing `IntegrationRegistry` singleton.

**Connection overhead:** stdio servers spawn a subprocess per connection (cheap, ~50ms). HTTP-transport servers keep a persistent TCP connection with session ID management.

---

## 3. Build vs Buy Comparison

| Factor | Composio (current) | MCP Catalog (own) | Hybrid |
|---|---|---|---|
| **Setup time** | One API key | 1–3 days per service server (Docker image, auth config, YAML entry) | Days for priority services; hours for rest |
| **Auth management** | Composio manages all OAuth flows and token refresh | You build/host token vault per-tenant | Own only for MCP servers; Composio for long tail |
| **Tool coverage** | 1000+ tools, 250+ apps | One server per service; only what you configure | MCP for 5–6 core; Composio for 900+ others |
| **Cost** | SaaS usage-based (opaque; likely $0.01–0.10/tool call at scale) | Infrastructure cost (containers + dev time) | Lower Composio spend + infra for MCP tier |
| **Reliability** | Composio uptime (SLA unknown publicly) | Your container SLA | Dual-path, higher resilience |
| **Vendor lock-in** | High — action slugs, auth flows, SDK all Composio-proprietary | Low — MCP is open protocol, swap servers freely | Medium |
| **Enterprise auth (SSO, per-tenant)** | Partial (they manage connected accounts per user_id) | Full control — you own OAuth flows, token storage, tenant isolation | Full for MCP services; Composio model for others |
| **Data residency** | All tool requests route through Composio servers (EU/US unclear) | Direct API calls — data never leaves your infra | Direct for MCP services |
| **Audit logging** | Composio's logs (limited visibility) | Full control — every JSON-RPC call loggable | Full for MCP tier |
| **Compliance (SOC2/GDPR)** | Depends on Composio's certifications; data shared with 3rd party | Self-hosted satisfies on-prem / data-residency requirements | Achievable for critical services |
| **Rate limits** | Composio-imposed (may be lower than service-native) | Service-native rate limits (often higher) | Service-native for MCP services |
| **Action coverage** | Composio's curated selection (may miss niche endpoints) | Full API surface of each service | Full for MCP services |
| **White-label capability** | No — Composio branding and endpoints visible in audit logs | Yes — completely transparent to enterprise clients | Yes for MCP tier |
| **Maintenance burden** | Minimal — Composio updates tool schemas | You track API changes, update server configs | Shared |
| **Migration risk** | Low (already working) | Medium — new infra, auth, testing needed | Low-medium |

---

## 4. B2B Enterprise Considerations

### 4.1 Data Residency (Critical for Enterprise Sales)

Composio routes all tool execution through their servers. This means when a worker creates a GitHub issue or searches Jira, the request payload — which may include proprietary code, ticket content, or customer data — is transmitted through Composio's infrastructure. For enterprise customers with strict data residency requirements (EU GDPR Article 44, FedRAMP, financial sector regulations), this is a **hard blocker**.

With direct MCP servers (HTTP transport, self-hosted), tool calls go directly from your Docker container to the target API. No third-party data routing. This is a compelling enterprise differentiator.

### 4.2 Per-Tenant OAuth Token Management

Composio handles per-user auth via their `user_id` parameter and connected account system (`composio add github`). It works but gives you no control over token lifecycle, rotation, or storage location.

For a multi-tenant B2B product, you need:
- A **token vault** (HashiCorp Vault, AWS Secrets Manager, or a simple encrypted Postgres table)
- Per-tenant OAuth flows (your app registers as the OAuth client with GitHub/Jira/etc.)
- Token refresh logic (the MCP spec doesn't prescribe this; you handle it in the MCPProvider layer)

This is the heaviest piece of the MCP migration. Estimate: 2–3 weeks to build a proper token vault and OAuth flow service.

### 4.3 On-Premise Deployment

MCP servers are just Docker containers. An enterprise customer with on-premise requirements can run the GitHub/Jira MCP servers inside their VPC alongside AiFactory. Composio has no on-prem offering.

### 4.4 Audit Logging

The MCP JSON-RPC protocol makes every tool call a discrete, loggable event:
```json
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_issue","arguments":{...}}}
```
This can be intercepted at the MCPClientPool level for structured audit logs with tool name, args, caller (worker ID), tenant, and timestamp. Composio audit logs (if available at all) are opaque.

### 4.5 Compliance Posture

| Requirement | Composio | MCP Catalog |
|---|---|---|
| SOC2 (depends on your cert) | You inherit Composio's risk | Within your audit scope |
| GDPR data processing agreement | Need Composio DPA | Not applicable (direct calls) |
| FedRAMP / GovCloud | Likely not achievable with Composio | Achievable (GovSlack MCP exists) |
| HIPAA | High bar — Composio BAA required | Achievable with proper controls |

---

## 5. Integration Architecture for Hybrid Approach

The existing `BaseIntegration` → `IntegrationRegistry` → `executor.py` architecture needs one new class:

```
integrations/
  base.py              (unchanged)
  registry.py          (unchanged)
  executor.py          (extend to handle mcp: tool prefix)
  providers/
    composio.py        (unchanged — fallback for long tail)
    mcp_provider.py    (NEW — wraps mcp.ClientSession per server)
config/integrations/
  mcp_catalog.yaml     (NEW — list of MCP servers per service)
```

**`MCPProvider`** would implement `BaseIntegration`, hold a dict of `ClientSession` objects keyed by service name, and route `execute_action(service::tool_name, params)` to the right session's `call_tool()`.

**Routing logic in executor.py**: The ACTION: block is extended or replaced. Two options:

1. **Extend ACTION: blocks** — workers write `ACTION:mcp\n{"server":"github","tool":"create_issue",...}\nACTION_END`. Minimal code change.
2. **Use OpenAI tool-call format** — execution lead injects tool schemas as LiteLLM `tools=` parameter; model emits structured tool calls; pipeline intercepts and routes. More robust; aligns with LiteLLM's native tool-use support. Higher refactor effort.

Option 1 is the lower-risk migration path.

---

## 6. Migration Effort Estimate

| Component | Composio → MCP Catalog | Effort |
|---|---|---|
| `mcp_provider.py` (new `BaseIntegration` subclass) | New file wrapping `mcp.ClientSession` | 2–3 days |
| `mcp_catalog.yaml` config (per-service server entries) | YAML with server URLs, auth env vars, tool prefixes | 1 day |
| OAuth token vault (per-tenant) | HashiCorp Vault or Postgres-backed token store | 2–3 weeks |
| Docker images for each MCP server | Dockerfile per service (most servers provide one) | 1–2 days |
| executor.py routing extension | Handle `mcp:server:tool` format | 1 day |
| Worker prompt changes | Replace Composio tool catalog injection with MCP `tools/list` | 1 day |
| Tests | Unit + integration for MCPProvider + executor routing | 3–5 days |
| Per-service auth setup (GitHub App, Jira OAuth App, etc.) | OAuth app registration at each service | 2–3 days |
| **Total (hybrid: 5 services)** | | **5–7 weeks** |
| **Total (full replacement)** | Add token vault + remaining 5 services + test coverage | **10–14 weeks** |

---

## 7. Recommendation

### Phase 1 (Now, 1–2 weeks): Prepare the abstraction layer

The `BaseIntegration` pattern is already correct. Add `MCPProvider` as a second provider that can be registered alongside Composio. No migration yet — just a new concrete class that wraps `mcp.ClientSession`. Gate it behind `AIFACTORY_MCP_CATALOG_ENABLED=false` by default.

### Phase 2 (Weeks 3–7): Migrate the 4 high-priority services to MCP

Migrate in this order, by ROI for enterprise sales:
1. **GitHub** — official server, 28.8k stars, zero auth risk (Personal Access Token or GitHub App). Immediate win.
2. **Atlassian (Jira + Confluence)** — 72 tools, Cloud + DataCenter support, critical for B2B enterprise. Use `sooperset/mcp-atlassian` or official Atlassian server.
3. **Notion** — official server, 22 tools, simple API-key auth.
4. **Google Workspace** — `taylorwilsdon/google_workspace_mcp` covers Gmail + Drive + Docs. Build token vault for OAuth before enabling.

Keep Composio active as fallback for all other services (Slack, Teams, Linear, 900+ long tail).

### Phase 3 (Months 3–6): Full MCP catalog + OAuth token vault

Build the per-tenant token vault. Add Slack, Teams, Linear MCP servers. Deprecate Composio dependency for production deployments. Composio can stay enabled for dev/quick-start mode.

### What NOT to do

- Do not rip out Composio before the token vault is built — OAuth management is the hardest part of this migration and the piece Composio currently handles for you.
- Do not use stdio transport for multi-tenant production — process-per-connection doesn't scale; use Streamable HTTP servers in containers.
- Do not try to build your own MCP servers for GitHub, Jira, or Notion — the official/community servers are good enough and actively maintained.

---

## 8. Summary Scorecard

| Criterion | Composio | MCP Catalog (Hybrid) | MCP Catalog (Full) |
|---|---|---|---|
| Time to first value | Immediate | 2–3 weeks | 10–14 weeks |
| Enterprise data residency | Fail | Pass for 4–5 services | Pass |
| On-premise support | No | Partial | Yes |
| Tool breadth | Excellent (1000+) | Good (5–6 services) | Good (10+ services) |
| Vendor dependency | High | Low (MCP is open) | None |
| Auth control | Poor | Full (for MCP tier) | Full |
| Audit/compliance | Poor | Good (MCP tier) | Excellent |
| Maintenance cost | Low | Medium | Medium-High |
| B2B enterprise positioning | Weak (routing through 3rd party) | Strong | Strongest |

**Verdict: Implement the hybrid approach now. The data residency and audit logging wins alone justify the effort for enterprise sales, and the phased migration keeps risk low.**
