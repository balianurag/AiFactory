# AiFactory Desktop App — Design Spec
_2026-04-23_

## Overview

A downloadable desktop application for AiFactory. Lightweight, fast, self-contained — users install one file and get the full pipeline experience without touching a terminal. Complements the existing web app and CLI; does not replace either.

---

## Architecture

### Repos

```
AiFactory/          ← Python backend (FastAPI) — unchanged
AiFactory-ui/       ← Next.js web app — unchanged, stays deployed
AiFactory-app/      ← NEW: Tauri + Vite + React desktop app
```

### AiFactory-app structure

```
AiFactory-app/
├── src-tauri/
│   ├── main.rs             # sidecar spawn/kill + Tauri IPC commands
│   ├── tauri.conf.json     # bundle config, sidecar declaration, window settings
│   └── binaries/           # PyInstaller-built aifactory-server sidecar
├── src/
│   ├── services/
│   │   └── api.ts          # copied from AiFactory-ui, NEXT_PUBLIC_ → VITE_
│   ├── components/         # ported from AiFactory-ui + new components
│   ├── pages/              # 8 view files
│   ├── App.tsx             # router + AppShell
│   └── main.tsx            # Vite entry
├── index.html
├── vite.config.ts
└── package.json
```

### Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Desktop shell | Tauri 2 (Rust) | ~5MB overhead vs ~150MB Electron; native OS integration |
| Frontend framework | Vite + React 18 + TypeScript | AiFactory-ui is already React; all components are client-only |
| Styling | Tailwind CSS | Already used in AiFactory-ui; copy config directly |
| Charts | recharts | Lightweight, React-native, no D3 dependency |
| Local storage | Tauri `store` plugin | API keys + settings stored encrypted on disk, never sent to backend |
| Backend | FastAPI sidecar (PyInstaller) | Same `aifactory serve` binary, bundled into the app |

### Sidecar lifecycle

1. App launches → Tauri `main.rs` spawns `aifactory-server` binary on `localhost:8080`
2. Frontend polls `/api/health` until ready (max 10s), shows loading screen
3. All HTTP and WebSocket traffic goes to `localhost:8080` — identical to web app
4. App quits → Tauri kills sidecar process
5. If sidecar crashes → frontend shows reconnect banner, Tauri attempts restart once

---

## Pages

### 1. Chat / Run
Core feature. Task input at bottom, streaming output in main area, agent spawn tree animating in a right panel as workers are created.

- Components: `TaskInput`, `MessageStream`, `AgentSpawnVisualizer`
- Endpoints: `POST /api/tasks`, `WS /api/ws/{task_id}`
- Streaming via WebSocket — tokens appear in real time
- Mode selector (best / balanced / cost_efficient) in input bar

### 2. Dashboard
Landing view after Chat. System health at a glance, quick action to start a new run.

- Components: `QuickStats`, `RecentRunsList`, `SystemStatusBar`
- Endpoints: `GET /api/tasks`, `GET /api/health`
- Stats: runs today, total tokens, avg cost per run, pipeline status

### 3. Cost & Usage Reports
New page — no equivalent in AiFactory-ui. Client-side aggregation from run history.

- Components: `DailySpendChart`, `ModelBreakdownChart`, `BudgetGauge`, `TokenSummaryTable`
- Endpoint: `GET /api/observability/runs` (paginated, all runs)
- Charts: daily spend line chart, per-model bar chart, token vs cost scatter
- Budget gauge shows current period spend vs `AIFACTORY_MAX_BUDGET_USD`

### 4. Run History
Browse and search past pipeline runs. Click any run to inspect full agent tree, outputs, and metrics.

- Components: `RunTable`, `RunDetailDrawer`, `AgentTree`, `SearchBar`
- Endpoints: `GET /api/observability/runs`, `GET /api/observability/runs/{run_id}`
- Search is client-side filter on loaded runs (no new backend endpoint needed)
- Run detail opens in a slide-over drawer, not a new page

### 5. MCP Catalog
Enable/disable built-in MCP servers and add custom ones. Persisted locally via Tauri store.

- Components: `CatalogGrid`, `ServerCard`, `AddMCPModal`
- Built-in servers shown from `config/integrations/mcp_catalog.yaml` via `GET /api/integrations`
- Toggle sends `POST /api/integrations/{name}/test` to verify before enabling
- Custom MCP: user provides URL + name + optional bearer token — stored in Tauri store, sent to backend as env override on next sidecar restart

### 6. Memory Explorer
Browse the 3-layer memory system with semantic search and layer-clear actions.

- Components: `MemoryLayerTabs`, `FactsTable`, `SearchInput`, `ClearLayerButton`
- Endpoints: `GET /api/memory/stats`, `POST /api/memory/retrieve`, `DELETE /api/memory/agent/{id}`

### 7. Model Registry
All models with 10 ability scores, RL performance stats, and tier assignments. Read-only in v1.

- Components: `ModelTable`, `AbilityScoreBars`, `RLStatsRow`, `TierBadge`
- Endpoint: `GET /api/models`

### 8. Settings
Local configuration. API keys encrypted at rest via Tauri store. Backend URL configurable for power users pointing to a remote server.

- Components: `KeyInput` (masked), `ModeSelector`, `BudgetCapInput`, `ToggleRow`
- All values stored in Tauri store — written to sidecar env on launch
- Backend URL field — default `localhost:8080`, overridable (disables sidecar auto-start)

---

## UI Design

- **Theme**: dark, near-black background (`#0a0a0a` / `#0f0f0f`), brick-red accent (`#e85d2c`)
- **Layout**: full sidebar (B), collapsible, logo + name always visible at top
- **Sidebar nav items**: Chat · Dashboard · Cost & Usage · Runs · MCP Catalog · Memory · Models · Settings
- **Typography**: system font stack for speed; monospace for code/output areas
- **Sidebar width**: 200px expanded, 48px collapsed (icon-only)

---

## Build Phases

| Phase | Scope | Est. |
|---|---|---|
| 1 — Scaffold | `AiFactory-app/` init, Tauri + Vite, `api.ts` port, health check, blank shell | 1 day |
| 2 — AppShell + Sidebar | Full sidebar, routing, theme, placeholder pages | 1 day |
| 3 — Chat / Run | TaskInput, MessageStream, AgentSpawnVisualizer, WebSocket streaming | 2 days |
| 4 — Dashboard + Run History | Dashboard stats, run table, run detail drawer | 1 day |
| 5 — Cost & Usage | New charts page, recharts, client-side aggregation | 1 day |
| 6 — MCP Catalog | Server grid, toggles, add custom MCP modal | 1 day |
| 7 — Memory + Models | Port memory/model views, wire endpoints | 1 day |
| 8 — Sidecar + Settings + Packaging | PyInstaller build, Tauri sidecar, Settings page, .dmg/.exe pipeline | 2 days |

**Total: ~10 days**

---

## Key Constraints

- Settings page writes to Tauri store — API keys never logged or sent to backend except as request headers
- Sidecar restart on crash is attempted once; on second failure the user is shown an error with a manual restart button
- Custom MCP servers take effect on next sidecar restart (not hot-reloaded in v1)
- Cost & Usage aggregates client-side — no new backend endpoint required
- Model Registry is read-only in v1 — editing ability scores deferred to v2
- If user sets a custom backend URL in Settings, sidecar auto-start is skipped entirely

---

## Out of Scope (v1)

- Auto-updater (Tauri updater plugin — v2)
- Multi-workspace / profile switching
- Editing model ability scores in UI
- Hot-reload of MCP config without restart
- Windows `.exe` packaging (macOS `.dmg` first, Windows in v2)
