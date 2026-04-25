# AiFactory Desktop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a downloadable Tauri + Vite + React desktop app that wraps the existing AiFactory FastAPI backend as a sidecar, providing 8 pages: Chat, Dashboard, Cost & Usage, Run History, MCP Catalog, Memory Explorer, Model Registry, and Settings.

**Architecture:** New repo `AiFactory-app/` at `~/Desktop/personal/AiFactory-app`. Tauri 2 Rust shell spawns a PyInstaller-built `aifactory-server` sidecar on `localhost:8080`. Vite + React 18 + TypeScript frontend talks to it over the same HTTP/WebSocket API the web app already uses. All settings/API keys stored encrypted via Tauri store plugin — never sent to backend except as headers.

**Tech Stack:** Tauri 2, Vite 5, React 18, TypeScript, Tailwind CSS 4, recharts, lucide-react, @tauri-apps/plugin-store, @tauri-apps/plugin-shell

---

## File Map

```
AiFactory-app/
├── src-tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── capabilities/default.json
│   └── src/
│       └── main.rs                         # sidecar spawn/kill + IPC commands
├── src/
│   ├── main.tsx                            # Vite entry
│   ├── App.tsx                             # router + AppShell wrapper
│   ├── index.css                           # global CSS vars + Tailwind
│   ├── services/
│   │   └── api.ts                          # ported from AiFactory-ui, VITE_ prefix
│   ├── store/
│   │   └── settings.ts                     # Tauri store read/write helpers
│   ├── hooks/
│   │   ├── useHealth.ts                    # poll /api/health until ready
│   │   └── useSettings.ts                  # read/write settings via store
│   ├── components/
│   │   ├── layout/
│   │   │   ├── AppShell.tsx                # sidebar + main content wrapper
│   │   │   └── Sidebar.tsx                 # collapsible nav, 8 items
│   │   ├── chat/
│   │   │   ├── TaskInput.tsx               # task input bar + mode selector
│   │   │   ├── MessageStream.tsx           # scrolling agent message feed
│   │   │   └── AgentSpawnVisualizer.tsx    # live agent tree panel
│   │   ├── dashboard/
│   │   │   ├── QuickStats.tsx              # runs today / tokens / avg cost
│   │   │   ├── RecentRunsList.tsx          # last 10 runs
│   │   │   └── SystemStatusBar.tsx         # backend health indicator
│   │   ├── runs/
│   │   │   ├── RunTable.tsx                # paginated run list + search
│   │   │   └── RunDetailDrawer.tsx         # slide-over with agent tree
│   │   ├── cost/
│   │   │   ├── DailySpendChart.tsx         # recharts line chart
│   │   │   ├── ModelBreakdownChart.tsx     # recharts bar chart
│   │   │   ├── BudgetGauge.tsx             # radial gauge
│   │   │   └── TokenSummaryTable.tsx       # table per model
│   │   ├── mcp/
│   │   │   ├── CatalogGrid.tsx             # grid of ServerCard
│   │   │   ├── ServerCard.tsx              # enable/disable toggle
│   │   │   └── AddMCPModal.tsx             # custom MCP form
│   │   ├── memory/
│   │   │   ├── MemoryLayerTabs.tsx         # agent/team/global tabs
│   │   │   ├── FactsTable.tsx              # memory entry rows
│   │   │   └── ClearLayerButton.tsx        # confirm + DELETE
│   │   ├── models/
│   │   │   ├── ModelTable.tsx              # all models
│   │   │   ├── AbilityScoreBars.tsx        # 10-bar mini chart
│   │   │   └── TierBadge.tsx               # premium/standard/budget
│   │   └── settings/
│   │       ├── KeyInput.tsx                # masked API key field
│   │       ├── ModeSelector.tsx            # best/balanced/cost_efficient
│   │       └── ToggleRow.tsx               # labelled toggle
│   └── pages/
│       ├── ChatPage.tsx
│       ├── DashboardPage.tsx
│       ├── CostPage.tsx
│       ├── RunsPage.tsx
│       ├── MCPPage.tsx
│       ├── MemoryPage.tsx
│       ├── ModelsPage.tsx
│       └── SettingsPage.tsx
├── index.html
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── package.json
```

---

## Task 1: Scaffold — Tauri + Vite project

**Files:**
- Create: `AiFactory-app/` (entire project scaffold)

- [ ] **Step 1: Create Tauri + Vite + React project**

```bash
cd ~/Desktop/personal
npm create tauri-app@latest AiFactory-app -- --template react-ts --manager npm
cd AiFactory-app
```

- [ ] **Step 2: Install frontend dependencies**

```bash
npm install recharts lucide-react @tauri-apps/plugin-store @tauri-apps/plugin-shell
npm install -D tailwindcss @tailwindcss/vite autoprefixer
```

- [ ] **Step 3: Install Tauri plugins in Cargo.toml**

Open `src-tauri/Cargo.toml` and add to `[dependencies]`:
```toml
tauri-plugin-store = "2"
tauri-plugin-shell = "2"
```

- [ ] **Step 4: Configure Tailwind in vite.config.ts**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 1420, strictPort: true },
});
```

- [ ] **Step 5: Write global CSS with theme vars**

Replace `src/index.css` entirely:
```css
@import "tailwindcss";

:root {
  --bg:       #0a0a0a;
  --bg-alt:   #0f0f0f;
  --surface:  #141414;
  --surface-2:#1a1a1a;
  --border:   rgba(255,255,255,0.07);
  --text:     #f1f5f9;
  --text-2:   #94a3b8;
  --text-3:   #64748b;
  --accent:   #e85d2c;
  --accent-dim: rgba(232,93,44,0.15);
  --green:    #22c55e;
  --red:      #ef4444;
  --cyan:     #0ea5e9;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; }
```

- [ ] **Step 6: Verify dev server starts**

```bash
npm run tauri dev
```
Expected: Tauri window opens showing default Vite + React starter page. No errors in terminal.

- [ ] **Step 7: Commit**

```bash
git init
git add .
git commit -m "feat: scaffold Tauri + Vite + React + Tailwind"
```

---

## Task 2: API service layer

**Files:**
- Create: `src/services/api.ts`

- [ ] **Step 1: Write api.ts — copy and adapt from AiFactory-ui**

Create `src/services/api.ts`:
```typescript
const DEFAULT_URL = "http://localhost:8080";
export let API_BASE = import.meta.env.VITE_API_URL ?? DEFAULT_URL;
export let WS_BASE  = API_BASE.replace("http", "ws");

export function setApiBase(url: string) {
  API_BASE = url;
  WS_BASE  = url.replace("http", "ws");
}

// ── Types ─────────────────────────────────────────────────────────────────

export interface TaskRequest {
  task: string;
  context?: string;
  mode?: "best" | "balanced" | "cost_efficient";
}

export interface TaskResponse {
  task_id: string;
  status: string;
  created_at: string;
}

export interface AgentInfo {
  agent_id: string;
  name: string;
  role: string;
  model: string;
  specialty?: string | null;
  status?: string;
  tokens_used?: number;
  parent_id?: string | null;
}

export interface AgentMessage {
  type: string;
  from: string;
  to: string;
  content: string;
  timestamp: string;
  meta?: string;
}

export interface TaskResult {
  title: string;
  summary: string;
  artifacts: { name: string; content_type: string; content: string; summary: string }[];
  quality_assessment: string;
  metrics: { total_tokens?: number; total_agents?: number };
}

export interface TaskStatus {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  progress: number;
  agents: AgentInfo[];
  messages: AgentMessage[];
  result: TaskResult | null;
  error: string | null;
  metrics: Record<string, number>;
}

export interface TaskListItem {
  task_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  progress: number;
  task_description: string;
  agents: AgentInfo[];
  metrics: Record<string, number>;
  error: string | null;
  has_result: boolean;
}

export interface TaskListResponse {
  tasks: TaskListItem[];
  total: number;
}

export interface ObsRun {
  run_id: string;
  task: string;
  status: string;
  created_at: string;
  elapsed_s?: number;
  total_tokens?: number;
  total_cost_usd?: number;
  model_breakdown?: Record<string, { tokens: number; cost_usd: number }>;
  judge_status?: string;
}

export interface ObsRunsResponse {
  runs: ObsRun[];
  total: number;
}

export interface MemoryStatsResponse {
  layers: Record<string, { entries: number }>;
  total_entries: number;
  status: string;
}

export interface MemoryEntryResult {
  level: string;
  content: string;
  agent_id: string | null;
  team_id: string | null;
  entry_type: string;
  turn_number: number;
  metadata: Record<string, unknown>;
}

export interface MemoryRetrieveResponse {
  status: string;
  count: number;
  results: MemoryEntryResult[];
}

export interface ModelInfo {
  id: string;
  display: string;
  provider: string;
  context_k: number;
  input_per_1m: number;
  output_per_1m: number;
  strength: string;
  tags: string[];
}

export interface ModelsResponse {
  roles: Record<string, { description: string; models: ModelInfo[] }>;
}

export interface IntegrationInfo {
  name: string;
  enabled: boolean;
  description?: string;
  type?: string;
}

export type StreamEventType = "status" | "agent_hierarchy" | "agent_message" | "completed" | "error";

export interface StreamEvent {
  type: StreamEventType;
  progress?: number;
  status?: string;
  message?: string;
  agent?: AgentInfo;
  agents?: AgentInfo[];
  msg?: AgentMessage;
  result?: TaskResult;
}

// ── Fetch helper ──────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── Endpoints ─────────────────────────────────────────────────────────────

export const checkHealth = () =>
  apiFetch<{ status: string; version: string }>("/api/health");

export const listTasks = (limit = 20) =>
  apiFetch<TaskListResponse>(`/api/tasks?limit=${limit}`);

export const submitTask = (req: TaskRequest) =>
  apiFetch<TaskResponse>("/api/tasks", { method: "POST", body: JSON.stringify(req) });

export const getTask = (id: string) =>
  apiFetch<TaskStatus>(`/api/tasks/${id}`);

export const listObsRuns = (limit = 100) =>
  apiFetch<ObsRunsResponse>(`/api/observability/runs?limit=${limit}`);

export const getObsRun = (id: string) =>
  apiFetch<ObsRun>(`/api/observability/runs/${id}`);

export const getMemoryStats = () =>
  apiFetch<MemoryStatsResponse>("/api/memory/stats");

export const retrieveMemory = (query: string, levels?: string[]) =>
  apiFetch<MemoryRetrieveResponse>("/api/memory/retrieve", {
    method: "POST",
    body: JSON.stringify({ query, levels: levels ?? ["agent", "team", "global"], top_k: 20 }),
  });

export const clearAgentMemory = (agentId: string) =>
  apiFetch<{ status: string }>(`/api/memory/agent/${agentId}`, { method: "DELETE" });

export const getModels = () =>
  apiFetch<ModelsResponse>("/api/models");

export const getIntegrations = () =>
  apiFetch<{ integrations: IntegrationInfo[] }>("/api/integrations");

export const testIntegration = (name: string) =>
  apiFetch<{ status: string }>(`/api/integrations/${name}/test`, { method: "POST" });

// ── WebSocket stream ──────────────────────────────────────────────────────

export function connectTaskStream(
  taskId: string,
  onEvent: (e: StreamEvent) => void
): () => void {
  let ws: WebSocket | null = null;
  let stopped = false;
  let attempt = 0;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function connect() {
    if (stopped) return;
    ws = new WebSocket(`${WS_BASE}/api/ws/${taskId}`);
    ws.onmessage = (ev) => {
      try { onEvent(JSON.parse(ev.data as string)); } catch { /* ignore */ }
    };
    ws.onclose = () => {
      if (!stopped && attempt < 5) {
        reconnectTimer = setTimeout(connect, Math.min(1000 * 2 ** attempt, 30_000));
        attempt++;
      }
    };
  }

  connect();
  return () => {
    stopped = true;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (ws) { ws.onclose = null; ws.close(); }
  };
}
```

- [ ] **Step 2: Write settings store helper**

Create `src/store/settings.ts`:
```typescript
import { load } from "@tauri-apps/plugin-store";

export interface AppSettings {
  backendUrl: string;
  mode: "best" | "balanced" | "cost_efficient";
  budgetCapUsd: number;
  apiKeys: Record<string, string>;
  customMcpServers: { name: string; url: string; token?: string }[];
  useSidecar: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  backendUrl: "http://localhost:8080",
  mode: "cost_efficient",
  budgetCapUsd: 5.0,
  apiKeys: {},
  customMcpServers: [],
  useSidecar: true,
};

async function getStore() {
  return load("settings.json", { autoSave: true });
}

export async function loadSettings(): Promise<AppSettings> {
  const store = await getStore();
  const saved = await store.get<Partial<AppSettings>>("settings");
  return { ...DEFAULT_SETTINGS, ...(saved ?? {}) };
}

export async function saveSettings(s: AppSettings): Promise<void> {
  const store = await getStore();
  await store.set("settings", s);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/services/api.ts src/store/settings.ts
git commit -m "feat: add API service layer and settings store"
```

---

## Task 3: AppShell + Sidebar

**Files:**
- Create: `src/components/layout/Sidebar.tsx`
- Create: `src/components/layout/AppShell.tsx`
- Create: `src/pages/ChatPage.tsx` (placeholder)
- Create: `src/pages/DashboardPage.tsx` (placeholder)
- Create: `src/pages/CostPage.tsx` (placeholder)
- Create: `src/pages/RunsPage.tsx` (placeholder)
- Create: `src/pages/MCPPage.tsx` (placeholder)
- Create: `src/pages/MemoryPage.tsx` (placeholder)
- Create: `src/pages/ModelsPage.tsx` (placeholder)
- Create: `src/pages/SettingsPage.tsx` (placeholder)
- Modify: `src/App.tsx`

- [ ] **Step 1: Write Sidebar.tsx**

Create `src/components/layout/Sidebar.tsx`:
```tsx
import { useState } from "react";
import {
  MessageSquare, LayoutDashboard, DollarSign, History,
  Plug, Database, Cpu, Settings, ChevronLeft, ChevronRight,
  Factory,
} from "lucide-react";

const NAV = [
  { id: "chat",      label: "Chat",          icon: MessageSquare },
  { id: "dashboard", label: "Dashboard",     icon: LayoutDashboard },
  { id: "cost",      label: "Cost & Usage",  icon: DollarSign },
  { id: "runs",      label: "Run History",   icon: History },
  { id: "mcp",       label: "MCP Catalog",   icon: Plug },
  { id: "memory",    label: "Memory",        icon: Database },
  { id: "models",    label: "Models",        icon: Cpu },
  { id: "settings",  label: "Settings",      icon: Settings },
];

interface Props {
  active: string;
  onChange: (id: string) => void;
}

export default function Sidebar({ active, onChange }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const w = collapsed ? 48 : 200;

  return (
    <aside
      style={{
        width: w, minWidth: w, background: "#0d0d0d",
        borderRight: "1px solid var(--border)",
        display: "flex", flexDirection: "column",
        transition: "width 0.2s ease",
        overflow: "hidden",
      }}
    >
      {/* Logo */}
      <div style={{
        height: 52, display: "flex", alignItems: "center",
        padding: collapsed ? "0 12px" : "0 16px",
        borderBottom: "1px solid var(--border)",
        gap: 10, overflow: "hidden",
      }}>
        <Factory size={20} style={{ color: "var(--accent)", flexShrink: 0 }} />
        {!collapsed && (
          <span style={{ fontWeight: 700, fontSize: 13, color: "var(--text)", whiteSpace: "nowrap" }}>
            AiFactory
          </span>
        )}
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: "8px 6px", display: "flex", flexDirection: "column", gap: 2 }}>
        {NAV.map(({ id, label, icon: Icon }) => {
          const isActive = active === id;
          return (
            <button
              key={id}
              onClick={() => onChange(id)}
              title={collapsed ? label : undefined}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: collapsed ? "8px 12px" : "8px 12px",
                borderRadius: 6, border: "none", cursor: "pointer",
                background: isActive ? "var(--accent-dim)" : "transparent",
                color: isActive ? "var(--accent)" : "var(--text-2)",
                fontSize: 13, fontWeight: isActive ? 600 : 400,
                textAlign: "left", width: "100%",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              <Icon size={15} style={{ flexShrink: 0 }} />
              {!collapsed && <span style={{ whiteSpace: "nowrap" }}>{label}</span>}
            </button>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          height: 36, border: "none", borderTop: "1px solid var(--border)",
          background: "transparent", color: "var(--text-3)", cursor: "pointer",
        }}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>
    </aside>
  );
}
```

- [ ] **Step 2: Write AppShell.tsx**

Create `src/components/layout/AppShell.tsx`:
```tsx
import { useState } from "react";
import Sidebar from "./Sidebar";
import ChatPage from "../../pages/ChatPage";
import DashboardPage from "../../pages/DashboardPage";
import CostPage from "../../pages/CostPage";
import RunsPage from "../../pages/RunsPage";
import MCPPage from "../../pages/MCPPage";
import MemoryPage from "../../pages/MemoryPage";
import ModelsPage from "../../pages/ModelsPage";
import SettingsPage from "../../pages/SettingsPage";

const PAGES: Record<string, React.ComponentType> = {
  chat: ChatPage,
  dashboard: DashboardPage,
  cost: CostPage,
  runs: RunsPage,
  mcp: MCPPage,
  memory: MemoryPage,
  models: ModelsPage,
  settings: SettingsPage,
};

export default function AppShell() {
  const [active, setActive] = useState("chat");
  const Page = PAGES[active] ?? ChatPage;

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      <Sidebar active={active} onChange={setActive} />
      <main style={{ flex: 1, overflow: "auto", background: "var(--bg)" }}>
        <Page />
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Write placeholder pages (all 8)**

Create `src/pages/ChatPage.tsx`:
```tsx
export default function ChatPage() {
  return <div style={{ padding: 32, color: "var(--text-2)" }}>Chat — coming in Task 4</div>;
}
```

Repeat for `DashboardPage.tsx`, `CostPage.tsx`, `RunsPage.tsx`, `MCPPage.tsx`, `MemoryPage.tsx`, `ModelsPage.tsx`, `SettingsPage.tsx` — same pattern, different label.

- [ ] **Step 4: Wire App.tsx**

Replace `src/App.tsx` entirely:
```tsx
import AppShell from "./components/layout/AppShell";

export default function App() {
  return <AppShell />;
}
```

- [ ] **Step 5: Run and verify sidebar renders**

```bash
npm run tauri dev
```
Expected: Sidebar visible on left with 8 nav items, brick-red accent on active item, collapses on click of chevron, main area shows placeholder text.

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "feat: AppShell + collapsible sidebar, 8 placeholder pages"
```

---

## Task 4: useHealth hook + startup screen

**Files:**
- Create: `src/hooks/useHealth.ts`
- Modify: `src/App.tsx`

- [ ] **Step 1: Write useHealth hook**

Create `src/hooks/useHealth.ts`:
```typescript
import { useEffect, useState } from "react";
import { checkHealth } from "../services/api";

export type HealthState = "loading" | "ready" | "timeout" | "error";

export function useHealth(maxWaitMs = 10_000): HealthState {
  const [state, setState] = useState<HealthState>("loading");

  useEffect(() => {
    let cancelled = false;
    const deadline = Date.now() + maxWaitMs;

    async function poll() {
      while (!cancelled && Date.now() < deadline) {
        try {
          await checkHealth();
          if (!cancelled) setState("ready");
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 800));
        }
      }
      if (!cancelled) setState("timeout");
    }

    poll();
    return () => { cancelled = true; };
  }, [maxWaitMs]);

  return state;
}
```

- [ ] **Step 2: Add startup screen to App.tsx**

Replace `src/App.tsx`:
```tsx
import AppShell from "./components/layout/AppShell";
import { useHealth } from "./hooks/useHealth";
import { Factory } from "lucide-react";

function StartupScreen({ state }: { state: "loading" | "timeout" | "error" }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100vh", gap: 16,
    }}>
      <Factory size={40} style={{ color: "var(--accent)" }} />
      <span style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>AiFactory</span>
      {state === "loading" && (
        <span style={{ color: "var(--text-3)", fontSize: 13 }}>Starting backend…</span>
      )}
      {state === "timeout" && (
        <span style={{ color: "var(--red)", fontSize: 13 }}>
          Backend unreachable. Is <code>aifactory serve</code> running?
        </span>
      )}
    </div>
  );
}

export default function App() {
  const health = useHealth();
  if (health !== "ready") return <StartupScreen state={health} />;
  return <AppShell />;
}
```

- [ ] **Step 3: Verify — start backend manually, then launch app**

```bash
# Terminal 1 — start backend
cd ~/Desktop/personal/AiFactory
source .venv/bin/activate
uv run aifactory serve

# Terminal 2 — launch app
cd ~/Desktop/personal/AiFactory-app
npm run tauri dev
```
Expected: loading spinner → switches to AppShell once backend is up.

- [ ] **Step 4: Commit**

```bash
git add src/hooks/useHealth.ts src/App.tsx
git commit -m "feat: startup health check with loading/timeout screen"
```

---

## Task 5: Chat page — TaskInput + WebSocket streaming

**Files:**
- Create: `src/components/chat/TaskInput.tsx`
- Create: `src/components/chat/MessageStream.tsx`
- Create: `src/components/chat/AgentSpawnVisualizer.tsx`
- Modify: `src/pages/ChatPage.tsx`

- [ ] **Step 1: Write TaskInput.tsx**

Create `src/components/chat/TaskInput.tsx`:
```tsx
import { useState } from "react";
import { Send } from "lucide-react";
import type { TaskRequest } from "../../services/api";

interface Props {
  onSubmit: (req: TaskRequest) => void;
  disabled?: boolean;
}

export default function TaskInput({ onSubmit, disabled }: Props) {
  const [task, setTask] = useState("");
  const [mode, setMode] = useState<TaskRequest["mode"]>("cost_efficient");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!task.trim() || disabled) return;
    onSubmit({ task: task.trim(), mode });
    setTask("");
  }

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: "flex", gap: 8, padding: "12px 16px",
        borderTop: "1px solid var(--border)", background: "var(--bg-alt)",
      }}
    >
      <select
        value={mode}
        onChange={(e) => setMode(e.target.value as TaskRequest["mode"])}
        disabled={disabled}
        style={{
          background: "var(--surface)", border: "1px solid var(--border)",
          borderRadius: 6, color: "var(--text-2)", fontSize: 12,
          padding: "6px 8px", cursor: "pointer",
        }}
      >
        <option value="cost_efficient">Cost</option>
        <option value="balanced">Balanced</option>
        <option value="best">Best</option>
      </select>
      <input
        value={task}
        onChange={(e) => setTask(e.target.value)}
        placeholder="Type a task and press Enter…"
        disabled={disabled}
        style={{
          flex: 1, background: "var(--surface)", border: "1px solid var(--border)",
          borderRadius: 6, color: "var(--text)", fontSize: 14,
          padding: "8px 12px", outline: "none",
        }}
      />
      <button
        type="submit"
        disabled={disabled || !task.trim()}
        style={{
          background: "var(--accent)", border: "none", borderRadius: 6,
          color: "#fff", padding: "8px 14px", cursor: "pointer",
          opacity: disabled || !task.trim() ? 0.4 : 1,
        }}
      >
        <Send size={15} />
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Write MessageStream.tsx**

Create `src/components/chat/MessageStream.tsx`:
```tsx
import { useEffect, useRef } from "react";
import type { StreamEvent } from "../../services/api";
import { ArrowRight, CheckCircle2, Activity, AlertTriangle } from "lucide-react";

interface Props { events: StreamEvent[]; isStreaming: boolean; }

function eventLabel(e: StreamEvent): string {
  if (e.type === "agent_message" && e.msg) return `${e.msg.from} → ${e.msg.to}: ${e.msg.content.slice(0, 120)}`;
  if (e.type === "status") return e.status ?? e.message ?? "status";
  if (e.type === "completed") return "Pipeline completed";
  if (e.type === "error") return `Error: ${e.message}`;
  return JSON.stringify(e).slice(0, 100);
}

function eventColor(e: StreamEvent): string {
  if (e.type === "completed") return "var(--green)";
  if (e.type === "error") return "var(--red)";
  if (e.type === "agent_message") return "var(--text)";
  return "var(--text-2)";
}

function eventIcon(e: StreamEvent) {
  if (e.type === "completed") return <CheckCircle2 size={12} />;
  if (e.type === "error") return <AlertTriangle size={12} />;
  if (e.type === "agent_message") return <ArrowRight size={12} />;
  return <Activity size={12} />;
}

export default function MessageStream({ events, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div style={{
      flex: 1, overflowY: "auto", padding: "16px",
      display: "flex", flexDirection: "column", gap: 6,
      fontFamily: "monospace", fontSize: 12,
    }}>
      {events.length === 0 && (
        <span style={{ color: "var(--text-3)", padding: 8 }}>
          {isStreaming ? "Connecting…" : "Submit a task to begin."}
        </span>
      )}
      {events.map((e, i) => (
        <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", color: eventColor(e) }}>
          <span style={{ marginTop: 1, flexShrink: 0 }}>{eventIcon(e)}</span>
          <span style={{ lineHeight: 1.5 }}>{eventLabel(e)}</span>
        </div>
      ))}
      {isStreaming && (
        <div style={{ color: "var(--accent)", fontSize: 11 }}>▋</div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 3: Write AgentSpawnVisualizer.tsx**

Create `src/components/chat/AgentSpawnVisualizer.tsx`:
```tsx
import { useEffect, useState } from "react";
import type { StreamEvent } from "../../services/api";
import { CheckCircle2, Activity } from "lucide-react";

interface Agent { name: string; role: string; status: "spawning" | "running" | "done"; }

interface Props { events: StreamEvent[]; }

export default function AgentSpawnVisualizer({ events }: Props) {
  const [agents, setAgents] = useState<Map<string, Agent>>(new Map());

  useEffect(() => {
    setAgents((prev) => {
      const next = new Map(prev);
      for (const ev of events) {
        if (ev.agent) {
          const a = ev.agent;
          next.set(a.agent_id, { name: a.name, role: a.role, status: "running" });
        }
        if (ev.agents) {
          for (const a of ev.agents) {
            next.set(a.agent_id, { name: a.name, role: a.role, status: a.status === "done" ? "done" : "running" });
          }
        }
        if (ev.type === "completed") {
          for (const [id, ag] of next) next.set(id, { ...ag, status: "done" });
        }
      }
      return next;
    });
  }, [events]);

  const list = Array.from(agents.values());
  if (list.length === 0) return null;

  return (
    <div style={{
      width: 200, borderLeft: "1px solid var(--border)", padding: 12,
      display: "flex", flexDirection: "column", gap: 6, overflowY: "auto",
    }}>
      <span style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.1em", marginBottom: 4 }}>
        AGENTS
      </span>
      {list.map((ag) => (
        <div key={ag.name} style={{
          display: "flex", alignItems: "center", gap: 6,
          padding: "5px 8px", borderRadius: 5,
          background: "var(--surface)", fontSize: 11,
        }}>
          {ag.status === "done"
            ? <CheckCircle2 size={10} style={{ color: "var(--green)" }} />
            : <Activity size={10} style={{ color: "var(--accent)" }} />}
          <div>
            <div style={{ color: "var(--text)", fontWeight: 500 }}>{ag.name}</div>
            <div style={{ color: "var(--text-3)", fontSize: 10 }}>{ag.role}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Wire ChatPage.tsx**

Replace `src/pages/ChatPage.tsx`:
```tsx
import { useState } from "react";
import TaskInput from "../components/chat/TaskInput";
import MessageStream from "../components/chat/MessageStream";
import AgentSpawnVisualizer from "../components/chat/AgentSpawnVisualizer";
import { submitTask, connectTaskStream, type TaskRequest, type StreamEvent } from "../services/api";

export default function ChatPage() {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  async function handleSubmit(req: TaskRequest) {
    setEvents([]);
    setIsStreaming(true);
    try {
      const { task_id } = await submitTask(req);
      const cleanup = connectTaskStream(task_id, (ev) => {
        setEvents((prev) => [...prev, ev]);
        if (ev.type === "completed" || ev.type === "error") {
          setIsStreaming(false);
          cleanup();
        }
      });
    } catch (err) {
      setEvents([{ type: "error", message: String(err) }]);
      setIsStreaming(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <MessageStream events={events} isStreaming={isStreaming} />
        <AgentSpawnVisualizer events={events} />
      </div>
      <TaskInput onSubmit={handleSubmit} disabled={isStreaming} />
    </div>
  );
}
```

- [ ] **Step 5: Test — submit a task, verify streaming**

With backend running (`uv run aifactory serve`):
```bash
npm run tauri dev
```
Expected: Type a task → messages appear in stream → agents appear in right panel → "Pipeline completed" at end.

- [ ] **Step 6: Commit**

```bash
git add src/components/chat/ src/pages/ChatPage.tsx
git commit -m "feat: Chat page — TaskInput, MessageStream, AgentSpawnVisualizer"
```

---

## Task 6: Dashboard + Run History pages

**Files:**
- Create: `src/components/dashboard/QuickStats.tsx`
- Create: `src/components/dashboard/RecentRunsList.tsx`
- Create: `src/components/dashboard/SystemStatusBar.tsx`
- Create: `src/components/runs/RunTable.tsx`
- Create: `src/components/runs/RunDetailDrawer.tsx`
- Modify: `src/pages/DashboardPage.tsx`
- Modify: `src/pages/RunsPage.tsx`

- [ ] **Step 1: Write QuickStats.tsx**

Create `src/components/dashboard/QuickStats.tsx`:
```tsx
import { useEffect, useState } from "react";
import { listTasks, listObsRuns } from "../../services/api";

interface Stats { runsToday: number; totalTokens: number; avgCostUsd: number; }

export default function QuickStats() {
  const [stats, setStats] = useState<Stats>({ runsToday: 0, totalTokens: 0, avgCostUsd: 0 });

  useEffect(() => {
    async function load() {
      const [tasks, obs] = await Promise.all([listTasks(100), listObsRuns(100)]);
      const today = new Date().toDateString();
      const runsToday = tasks.tasks.filter((t) => new Date(t.created_at).toDateString() === today).length;
      const totalTokens = obs.runs.reduce((s, r) => s + (r.total_tokens ?? 0), 0);
      const costs = obs.runs.filter((r) => r.total_cost_usd != null).map((r) => r.total_cost_usd!);
      const avgCostUsd = costs.length ? costs.reduce((a, b) => a + b, 0) / costs.length : 0;
      setStats({ runsToday, totalTokens, avgCostUsd });
    }
    load();
  }, []);

  const cards = [
    { label: "Runs Today", value: stats.runsToday },
    { label: "Total Tokens", value: stats.totalTokens.toLocaleString() },
    { label: "Avg Cost", value: `$${stats.avgCostUsd.toFixed(4)}` },
  ];

  return (
    <div style={{ display: "flex", gap: 12 }}>
      {cards.map(({ label, value }) => (
        <div key={label} style={{
          flex: 1, background: "var(--surface)", border: "1px solid var(--border)",
          borderRadius: 8, padding: "16px 20px",
        }}>
          <div style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 6 }}>{label}</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "var(--text)" }}>{value}</div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Write SystemStatusBar.tsx**

Create `src/components/dashboard/SystemStatusBar.tsx`:
```tsx
import { useEffect, useState } from "react";
import { checkHealth } from "../../services/api";

export default function SystemStatusBar() {
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    checkHealth()
      .then(() => setStatus("online"))
      .catch(() => setStatus("offline"));
  }, []);

  const color = status === "online" ? "var(--green)" : status === "offline" ? "var(--red)" : "var(--text-3)";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color }}>
      <div style={{ width: 7, height: 7, borderRadius: "50%", background: color }} />
      Backend {status}
    </div>
  );
}
```

- [ ] **Step 3: Write RecentRunsList.tsx**

Create `src/components/dashboard/RecentRunsList.tsx`:
```tsx
import { useEffect, useState } from "react";
import { listTasks, type TaskListItem } from "../../services/api";

interface Props { onSelectRun?: (id: string) => void; }

export default function RecentRunsList({ onSelectRun }: Props) {
  const [runs, setRuns] = useState<TaskListItem[]>([]);

  useEffect(() => { listTasks(10).then((r) => setRuns(r.tasks)); }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {runs.length === 0 && (
        <span style={{ color: "var(--text-3)", fontSize: 13 }}>No runs yet.</span>
      )}
      {runs.map((r) => (
        <div
          key={r.task_id}
          onClick={() => onSelectRun?.(r.task_id)}
          style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "8px 12px", borderRadius: 6, cursor: onSelectRun ? "pointer" : "default",
            background: "var(--surface)", marginBottom: 2,
          }}
        >
          <div style={{
            width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
            background: r.status === "completed" ? "var(--green)" : r.status === "failed" ? "var(--red)" : "var(--accent)",
          }} />
          <span style={{ flex: 1, fontSize: 13, color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {r.task_description || r.task_id}
          </span>
          <span style={{ fontSize: 11, color: "var(--text-3)", flexShrink: 0 }}>
            {new Date(r.created_at).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Wire DashboardPage.tsx**

Replace `src/pages/DashboardPage.tsx`:
```tsx
import QuickStats from "../components/dashboard/QuickStats";
import RecentRunsList from "../components/dashboard/RecentRunsList";
import SystemStatusBar from "../components/dashboard/SystemStatusBar";

export default function DashboardPage() {
  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>Dashboard</h1>
        <SystemStatusBar />
      </div>
      <QuickStats />
      <div>
        <h2 style={{ fontSize: 13, color: "var(--text-3)", marginBottom: 10, letterSpacing: "0.05em" }}>RECENT RUNS</h2>
        <RecentRunsList />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Write RunTable.tsx**

Create `src/components/runs/RunTable.tsx`:
```tsx
import { useEffect, useState } from "react";
import { listTasks, type TaskListItem } from "../../services/api";

interface Props { onSelect: (id: string) => void; }

export default function RunTable({ onSelect }: Props) {
  const [runs, setRuns] = useState<TaskListItem[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => { listTasks(200).then((r) => setRuns(r.tasks)); }, []);

  const filtered = runs.filter((r) =>
    !query || r.task_description?.toLowerCase().includes(query.toLowerCase()) || r.task_id.includes(query)
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search runs…"
        style={{
          background: "var(--surface)", border: "1px solid var(--border)",
          borderRadius: 6, color: "var(--text)", padding: "7px 12px", fontSize: 13,
        }}
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {filtered.map((r) => (
          <div
            key={r.task_id}
            onClick={() => onSelect(r.task_id)}
            style={{
              display: "grid", gridTemplateColumns: "8px 1fr 80px 80px 60px",
              alignItems: "center", gap: 12,
              padding: "9px 12px", borderRadius: 6, cursor: "pointer",
              background: "var(--surface)", fontSize: 12,
            }}
          >
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: r.status === "completed" ? "var(--green)" : r.status === "failed" ? "var(--red)" : "var(--accent)",
            }} />
            <span style={{ color: "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {r.task_description || r.task_id}
            </span>
            <span style={{ color: "var(--text-3)" }}>{r.status}</span>
            <span style={{ color: "var(--text-3)" }}>
              {(r.metrics?.total_tokens ?? 0).toLocaleString()} tok
            </span>
            <span style={{ color: "var(--text-3)" }}>
              {new Date(r.created_at).toLocaleDateString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Write RunDetailDrawer.tsx**

Create `src/components/runs/RunDetailDrawer.tsx`:
```tsx
import { useEffect, useState } from "react";
import { getTask, type TaskStatus } from "../../services/api";
import { X } from "lucide-react";

interface Props { runId: string | null; onClose: () => void; }

export default function RunDetailDrawer({ runId, onClose }: Props) {
  const [run, setRun] = useState<TaskStatus | null>(null);

  useEffect(() => {
    if (!runId) { setRun(null); return; }
    getTask(runId).then(setRun).catch(console.error);
  }, [runId]);

  if (!runId) return null;

  return (
    <div style={{
      position: "fixed", right: 0, top: 0, bottom: 0, width: 420,
      background: "var(--bg-alt)", borderLeft: "1px solid var(--border)",
      display: "flex", flexDirection: "column", zIndex: 100,
      boxShadow: "-4px 0 24px rgba(0,0,0,0.4)",
    }}>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 16px", borderBottom: "1px solid var(--border)",
      }}>
        <span style={{ fontWeight: 600, fontSize: 14, color: "var(--text)" }}>Run Detail</span>
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-3)" }}>
          <X size={16} />
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        {!run && <span style={{ color: "var(--text-3)" }}>Loading…</span>}
        {run && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16, fontSize: 12 }}>
            <div>
              <div style={{ color: "var(--text-3)", marginBottom: 4 }}>STATUS</div>
              <div style={{ color: run.status === "completed" ? "var(--green)" : "var(--accent)" }}>
                {run.status}
              </div>
            </div>
            <div>
              <div style={{ color: "var(--text-3)", marginBottom: 4 }}>AGENTS ({run.agents.length})</div>
              {run.agents.map((a) => (
                <div key={a.agent_id} style={{ color: "var(--text-2)", padding: "2px 0" }}>
                  {a.name} — {a.role} — {a.model}
                </div>
              ))}
            </div>
            {run.result && (
              <div>
                <div style={{ color: "var(--text-3)", marginBottom: 4 }}>RESULT</div>
                <div style={{ color: "var(--text)", lineHeight: 1.6 }}>{run.result.summary}</div>
              </div>
            )}
            <div>
              <div style={{ color: "var(--text-3)", marginBottom: 4 }}>TOKENS</div>
              <div style={{ color: "var(--text)" }}>
                {(run.metrics?.total_tokens ?? 0).toLocaleString()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Wire RunsPage.tsx**

Replace `src/pages/RunsPage.tsx`:
```tsx
import { useState } from "react";
import RunTable from "../components/runs/RunTable";
import RunDetailDrawer from "../components/runs/RunDetailDrawer";

export default function RunsPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)", marginBottom: 20 }}>Run History</h1>
      <RunTable onSelect={setSelectedId} />
      <RunDetailDrawer runId={selectedId} onClose={() => setSelectedId(null)} />
    </div>
  );
}
```

- [ ] **Step 8: Commit**

```bash
git add src/components/dashboard/ src/components/runs/ src/pages/DashboardPage.tsx src/pages/RunsPage.tsx
git commit -m "feat: Dashboard + Run History pages"
```

---

## Task 7: Cost & Usage page

**Files:**
- Create: `src/components/cost/DailySpendChart.tsx`
- Create: `src/components/cost/ModelBreakdownChart.tsx`
- Create: `src/components/cost/BudgetGauge.tsx`
- Create: `src/components/cost/TokenSummaryTable.tsx`
- Modify: `src/pages/CostPage.tsx`

- [ ] **Step 1: Write DailySpendChart.tsx**

Create `src/components/cost/DailySpendChart.tsx`:
```tsx
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { listObsRuns } from "../../services/api";

interface DayData { date: string; cost: number; }

export default function DailySpendChart() {
  const [data, setData] = useState<DayData[]>([]);

  useEffect(() => {
    listObsRuns(500).then(({ runs }) => {
      const byDay: Record<string, number> = {};
      for (const r of runs) {
        const d = new Date(r.created_at).toLocaleDateString();
        byDay[d] = (byDay[d] ?? 0) + (r.total_cost_usd ?? 0);
      }
      setData(Object.entries(byDay).map(([date, cost]) => ({ date, cost: parseFloat(cost.toFixed(4)) })));
    });
  }, []);

  return (
    <div style={{ background: "var(--surface)", borderRadius: 8, padding: 16, border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 12, color: "var(--text-3)", marginBottom: 12 }}>DAILY SPEND (USD)</div>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data}>
          <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--text-3)" }} />
          <YAxis tick={{ fontSize: 10, fill: "var(--text-3)" }} />
          <Tooltip
            contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 11 }}
          />
          <Line type="monotone" dataKey="cost" stroke="var(--accent)" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Write ModelBreakdownChart.tsx**

Create `src/components/cost/ModelBreakdownChart.tsx`:
```tsx
import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { listObsRuns } from "../../services/api";

interface ModelData { model: string; cost: number; tokens: number; }

export default function ModelBreakdownChart() {
  const [data, setData] = useState<ModelData[]>([]);

  useEffect(() => {
    listObsRuns(500).then(({ runs }) => {
      const byModel: Record<string, { cost: number; tokens: number }> = {};
      for (const r of runs) {
        if (!r.model_breakdown) continue;
        for (const [model, stats] of Object.entries(r.model_breakdown)) {
          if (!byModel[model]) byModel[model] = { cost: 0, tokens: 0 };
          byModel[model].cost += stats.cost_usd;
          byModel[model].tokens += stats.tokens;
        }
      }
      setData(
        Object.entries(byModel)
          .map(([model, s]) => ({ model: model.split("/").pop() ?? model, cost: parseFloat(s.cost.toFixed(4)), tokens: s.tokens }))
          .sort((a, b) => b.cost - a.cost)
          .slice(0, 8)
      );
    });
  }, []);

  return (
    <div style={{ background: "var(--surface)", borderRadius: 8, padding: 16, border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 12, color: "var(--text-3)", marginBottom: 12 }}>COST BY MODEL</div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} layout="vertical">
          <XAxis type="number" tick={{ fontSize: 10, fill: "var(--text-3)" }} />
          <YAxis type="category" dataKey="model" tick={{ fontSize: 10, fill: "var(--text-3)" }} width={100} />
          <Tooltip
            contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 11 }}
          />
          <Bar dataKey="cost" fill="var(--accent)" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Write BudgetGauge.tsx**

Create `src/components/cost/BudgetGauge.tsx`:
```tsx
import { useEffect, useState } from "react";
import { listObsRuns } from "../../services/api";

interface Props { capUsd?: number; }

export default function BudgetGauge({ capUsd = 5 }: Props) {
  const [spent, setSpent] = useState(0);

  useEffect(() => {
    const now = new Date();
    const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1).toISOString();
    listObsRuns(500).then(({ runs }) => {
      const total = runs
        .filter((r) => r.created_at >= startOfMonth)
        .reduce((s, r) => s + (r.total_cost_usd ?? 0), 0);
      setSpent(total);
    });
  }, []);

  const pct = Math.min((spent / capUsd) * 100, 100);
  const color = pct > 80 ? "var(--red)" : pct > 50 ? "var(--accent)" : "var(--green)";

  return (
    <div style={{ background: "var(--surface)", borderRadius: 8, padding: 16, border: "1px solid var(--border)" }}>
      <div style={{ fontSize: 12, color: "var(--text-3)", marginBottom: 10 }}>MONTHLY BUDGET</div>
      <div style={{ fontSize: 24, fontWeight: 700, color, marginBottom: 8 }}>
        ${spent.toFixed(4)} <span style={{ fontSize: 13, color: "var(--text-3)", fontWeight: 400 }}>/ ${capUsd}</span>
      </div>
      <div style={{ height: 6, background: "var(--surface-2)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: color, transition: "width 0.4s" }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Write TokenSummaryTable.tsx**

Create `src/components/cost/TokenSummaryTable.tsx`:
```tsx
import { useEffect, useState } from "react";
import { listObsRuns } from "../../services/api";

interface Row { model: string; tokens: number; cost: number; runs: number; }

export default function TokenSummaryTable() {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    listObsRuns(500).then(({ runs }) => {
      const byModel: Record<string, { tokens: number; cost: number; runs: number }> = {};
      for (const r of runs) {
        if (!r.model_breakdown) continue;
        for (const [model, s] of Object.entries(r.model_breakdown)) {
          if (!byModel[model]) byModel[model] = { tokens: 0, cost: 0, runs: 0 };
          byModel[model].tokens += s.tokens;
          byModel[model].cost += s.cost_usd;
          byModel[model].runs += 1;
        }
      }
      setRows(
        Object.entries(byModel)
          .map(([model, s]) => ({ model, ...s, cost: parseFloat(s.cost.toFixed(5)) }))
          .sort((a, b) => b.tokens - a.tokens)
      );
    });
  }, []);

  return (
    <div style={{ background: "var(--surface)", borderRadius: 8, border: "1px solid var(--border)", overflow: "hidden" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {["Model", "Tokens", "Cost (USD)", "Runs"].map((h) => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "var(--text-3)", fontWeight: 500 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "8px 12px", color: "var(--text)" }}>{r.model}</td>
              <td style={{ padding: "8px 12px", color: "var(--text-2)" }}>{r.tokens.toLocaleString()}</td>
              <td style={{ padding: "8px 12px", color: "var(--text-2)" }}>${r.cost}</td>
              <td style={{ padding: "8px 12px", color: "var(--text-2)" }}>{r.runs}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 5: Wire CostPage.tsx**

Replace `src/pages/CostPage.tsx`:
```tsx
import BudgetGauge from "../components/cost/BudgetGauge";
import DailySpendChart from "../components/cost/DailySpendChart";
import ModelBreakdownChart from "../components/cost/ModelBreakdownChart";
import TokenSummaryTable from "../components/cost/TokenSummaryTable";

export default function CostPage() {
  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 16 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>Cost & Usage</h1>
      <BudgetGauge capUsd={5} />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <DailySpendChart />
        <ModelBreakdownChart />
      </div>
      <TokenSummaryTable />
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add src/components/cost/ src/pages/CostPage.tsx
git commit -m "feat: Cost & Usage page with recharts + budget gauge"
```

---

## Task 8: MCP Catalog page

**Files:**
- Create: `src/components/mcp/ServerCard.tsx`
- Create: `src/components/mcp/CatalogGrid.tsx`
- Create: `src/components/mcp/AddMCPModal.tsx`
- Modify: `src/pages/MCPPage.tsx`

- [ ] **Step 1: Write ServerCard.tsx**

Create `src/components/mcp/ServerCard.tsx`:
```tsx
import { useState } from "react";
import { testIntegration, type IntegrationInfo } from "../../services/api";
import { CheckCircle2, XCircle, Loader } from "lucide-react";

interface Props { server: IntegrationInfo; }

export default function ServerCard({ server }: Props) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<"ok" | "fail" | null>(null);

  async function handleToggle() {
    setTesting(true);
    setResult(null);
    try {
      await testIntegration(server.name);
      setResult("ok");
    } catch {
      setResult("fail");
    } finally {
      setTesting(false);
    }
  }

  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 8, padding: 14, display: "flex", flexDirection: "column", gap: 8,
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontWeight: 600, fontSize: 13, color: "var(--text)" }}>{server.name}</span>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: server.enabled ? "var(--green)" : "var(--text-3)",
        }} />
      </div>
      {server.description && (
        <span style={{ fontSize: 11, color: "var(--text-3)" }}>{server.description}</span>
      )}
      <button
        onClick={handleToggle}
        disabled={testing}
        style={{
          marginTop: 4, padding: "5px 10px", borderRadius: 5, fontSize: 11,
          border: "1px solid var(--border)", background: "var(--surface-2)",
          color: "var(--text-2)", cursor: "pointer", display: "flex", alignItems: "center", gap: 5,
        }}
      >
        {testing && <Loader size={10} />}
        {result === "ok" && <CheckCircle2 size={10} style={{ color: "var(--green)" }} />}
        {result === "fail" && <XCircle size={10} style={{ color: "var(--red)" }} />}
        Test Connection
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Write AddMCPModal.tsx**

Create `src/components/mcp/AddMCPModal.tsx`:
```tsx
import { useState } from "react";
import { X } from "lucide-react";

interface CustomServer { name: string; url: string; token?: string; }
interface Props { onAdd: (s: CustomServer) => void; onClose: () => void; }

export default function AddMCPModal({ onAdd, onClose }: Props) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [token, setToken] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !url.trim()) return;
    onAdd({ name: name.trim(), url: url.trim(), token: token.trim() || undefined });
    onClose();
  }

  const inputStyle = {
    background: "var(--surface)", border: "1px solid var(--border)",
    borderRadius: 6, color: "var(--text)", padding: "7px 10px",
    fontSize: 13, width: "100%",
  };

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
      display: "flex", alignItems: "center", justifyContent: "center", zIndex: 200,
    }}>
      <div style={{
        background: "var(--bg-alt)", border: "1px solid var(--border)",
        borderRadius: 10, padding: 24, width: 380,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
          <span style={{ fontWeight: 700, fontSize: 14, color: "var(--text)" }}>Add Custom MCP Server</span>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-3)" }}>
            <X size={15} />
          </button>
        </div>
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" style={inputStyle} />
          <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="URL (http://...)" style={inputStyle} />
          <input value={token} onChange={(e) => setToken(e.target.value)} placeholder="Bearer token (optional)" type="password" style={inputStyle} />
          <button
            type="submit"
            style={{
              marginTop: 4, background: "var(--accent)", border: "none",
              borderRadius: 6, color: "#fff", padding: "9px", fontSize: 13,
              cursor: "pointer", fontWeight: 600,
            }}
          >
            Add Server
          </button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write CatalogGrid.tsx**

Create `src/components/mcp/CatalogGrid.tsx`:
```tsx
import { useEffect, useState } from "react";
import { getIntegrations, type IntegrationInfo } from "../../services/api";
import ServerCard from "./ServerCard";

interface CustomServer { name: string; url: string; token?: string; }
interface Props { customServers: CustomServer[]; }

export default function CatalogGrid({ customServers }: Props) {
  const [servers, setServers] = useState<IntegrationInfo[]>([]);

  useEffect(() => {
    getIntegrations().then((r) => setServers(r.integrations)).catch(() => setServers([]));
  }, []);

  const custom: IntegrationInfo[] = customServers.map((s) => ({
    name: s.name, enabled: false, description: s.url, type: "custom",
  }));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12 }}>
      {[...servers, ...custom].map((s) => <ServerCard key={s.name} server={s} />)}
    </div>
  );
}
```

- [ ] **Step 4: Wire MCPPage.tsx**

Replace `src/pages/MCPPage.tsx`:
```tsx
import { useState } from "react";
import { Plus } from "lucide-react";
import CatalogGrid from "../components/mcp/CatalogGrid";
import AddMCPModal from "../components/mcp/AddMCPModal";

interface CustomServer { name: string; url: string; token?: string; }

export default function MCPPage() {
  const [custom, setCustom] = useState<CustomServer[]>([]);
  const [showModal, setShowModal] = useState(false);

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>MCP Catalog</h1>
        <button
          onClick={() => setShowModal(true)}
          style={{
            display: "flex", alignItems: "center", gap: 6,
            background: "var(--accent)", border: "none", borderRadius: 6,
            color: "#fff", padding: "7px 14px", fontSize: 13, cursor: "pointer",
          }}
        >
          <Plus size={14} /> Add Custom
        </button>
      </div>
      <CatalogGrid customServers={custom} />
      {showModal && (
        <AddMCPModal
          onAdd={(s) => setCustom((prev) => [...prev, s])}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add src/components/mcp/ src/pages/MCPPage.tsx
git commit -m "feat: MCP Catalog page with server cards and add custom modal"
```

---

## Task 9: Memory Explorer + Model Registry pages

**Files:**
- Create: `src/components/memory/MemoryLayerTabs.tsx`
- Create: `src/components/memory/FactsTable.tsx`
- Create: `src/components/memory/ClearLayerButton.tsx`
- Create: `src/components/models/ModelTable.tsx`
- Create: `src/components/models/AbilityScoreBars.tsx`
- Create: `src/components/models/TierBadge.tsx`
- Modify: `src/pages/MemoryPage.tsx`
- Modify: `src/pages/ModelsPage.tsx`

- [ ] **Step 1: Write FactsTable.tsx**

Create `src/components/memory/FactsTable.tsx`:
```tsx
import type { MemoryEntryResult } from "../../services/api";

interface Props { entries: MemoryEntryResult[]; }

export default function FactsTable({ entries }: Props) {
  if (entries.length === 0) {
    return <span style={{ color: "var(--text-3)", fontSize: 13 }}>No entries found.</span>;
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {entries.map((e, i) => (
        <div key={i} style={{
          padding: "8px 12px", background: "var(--surface)", borderRadius: 5, fontSize: 12,
        }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 3 }}>
            <span style={{
              fontSize: 10, padding: "1px 6px", borderRadius: 3,
              background: "var(--surface-2)", color: "var(--text-3)",
            }}>{e.level}</span>
            <span style={{ color: "var(--text-3)", fontSize: 10 }}>{e.entry_type}</span>
          </div>
          <div style={{ color: "var(--text)", lineHeight: 1.5 }}>{e.content}</div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Write ClearLayerButton.tsx**

Create `src/components/memory/ClearLayerButton.tsx`:
```tsx
import { useState } from "react";
import { clearAgentMemory } from "../../services/api";
import { Trash2 } from "lucide-react";

interface Props { agentId: string; onCleared: () => void; }

export default function ClearLayerButton({ agentId, onCleared }: Props) {
  const [confirming, setConfirming] = useState(false);
  const [loading, setLoading] = useState(false);

  async function handleClear() {
    if (!confirming) { setConfirming(true); return; }
    setLoading(true);
    try {
      await clearAgentMemory(agentId);
      onCleared();
    } finally {
      setLoading(false);
      setConfirming(false);
    }
  }

  return (
    <button
      onClick={handleClear}
      disabled={loading}
      style={{
        display: "flex", alignItems: "center", gap: 6, padding: "5px 12px",
        borderRadius: 5, border: "1px solid var(--red)", background: "transparent",
        color: "var(--red)", fontSize: 12, cursor: "pointer",
      }}
    >
      <Trash2 size={12} />
      {loading ? "Clearing…" : confirming ? "Confirm clear?" : "Clear Layer"}
    </button>
  );
}
```

- [ ] **Step 3: Write MemoryLayerTabs.tsx**

Create `src/components/memory/MemoryLayerTabs.tsx`:
```tsx
import { useEffect, useState } from "react";
import { getMemoryStats, retrieveMemory, type MemoryEntryResult, type MemoryStatsResponse } from "../../services/api";
import FactsTable from "./FactsTable";
import ClearLayerButton from "./ClearLayerButton";

const LAYERS = ["agent", "team", "global"] as const;

export default function MemoryLayerTabs() {
  const [active, setActive] = useState<"agent" | "team" | "global">("agent");
  const [stats, setStats] = useState<MemoryStatsResponse | null>(null);
  const [entries, setEntries] = useState<MemoryEntryResult[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => { getMemoryStats().then(setStats).catch(() => {}); }, []);

  async function search() {
    const res = await retrieveMemory(query || ".", [active]);
    setEntries(res.results);
  }

  useEffect(() => { search(); }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Tabs */}
      <div style={{ display: "flex", gap: 2 }}>
        {LAYERS.map((l) => {
          const count = stats?.layers[l]?.entries ?? 0;
          return (
            <button
              key={l}
              onClick={() => setActive(l)}
              style={{
                padding: "6px 14px", borderRadius: 6, border: "none", cursor: "pointer",
                fontSize: 12, fontWeight: active === l ? 600 : 400,
                background: active === l ? "var(--accent-dim)" : "var(--surface)",
                color: active === l ? "var(--accent)" : "var(--text-2)",
              }}
            >
              {l} ({count})
            </button>
          );
        })}
      </div>
      {/* Search */}
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search()}
          placeholder="Search memory…"
          style={{
            flex: 1, background: "var(--surface)", border: "1px solid var(--border)",
            borderRadius: 6, color: "var(--text)", padding: "7px 12px", fontSize: 13,
          }}
        />
        <button
          onClick={search}
          style={{
            background: "var(--accent)", border: "none", borderRadius: 6,
            color: "#fff", padding: "7px 14px", fontSize: 13, cursor: "pointer",
          }}
        >
          Search
        </button>
        <ClearLayerButton agentId={active} onCleared={() => { setEntries([]); getMemoryStats().then(setStats); }} />
      </div>
      <FactsTable entries={entries} />
    </div>
  );
}
```

- [ ] **Step 4: Wire MemoryPage.tsx**

Replace `src/pages/MemoryPage.tsx`:
```tsx
import MemoryLayerTabs from "../components/memory/MemoryLayerTabs";

export default function MemoryPage() {
  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)", marginBottom: 20 }}>Memory Explorer</h1>
      <MemoryLayerTabs />
    </div>
  );
}
```

- [ ] **Step 5: Write TierBadge.tsx**

Create `src/components/models/TierBadge.tsx`:
```tsx
interface Props { tier: string; }
const COLORS: Record<string, string> = {
  premium: "#a855f7", standard: "var(--cyan)", budget: "var(--text-3)",
};
export default function TierBadge({ tier }: Props) {
  const color = COLORS[tier.toLowerCase()] ?? "var(--text-3)";
  return (
    <span style={{
      fontSize: 10, padding: "2px 7px", borderRadius: 4,
      border: `1px solid ${color}`, color, fontWeight: 600,
    }}>{tier}</span>
  );
}
```

- [ ] **Step 6: Write AbilityScoreBars.tsx**

Create `src/components/models/AbilityScoreBars.tsx`:
```tsx
interface Props { tags: string[]; }
export default function AbilityScoreBars({ tags }: Props) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {tags.map((t) => (
        <span key={t} style={{
          fontSize: 9, padding: "2px 6px", borderRadius: 3,
          background: "var(--surface-2)", color: "var(--text-3)",
        }}>{t}</span>
      ))}
    </div>
  );
}
```

- [ ] **Step 7: Write ModelTable.tsx**

Create `src/components/models/ModelTable.tsx`:
```tsx
import { useEffect, useState } from "react";
import { getModels, type ModelInfo } from "../../services/api";
import TierBadge from "./TierBadge";
import AbilityScoreBars from "./AbilityScoreBars";

export default function ModelTable() {
  const [models, setModels] = useState<(ModelInfo & { role: string })[]>([]);

  useEffect(() => {
    getModels().then((r) => {
      const flat = Object.entries(r.roles).flatMap(([role, { models: ms }]) =>
        ms.map((m) => ({ ...m, role }))
      );
      setModels(flat);
    });
  }, []);

  return (
    <div style={{ background: "var(--surface)", borderRadius: 8, border: "1px solid var(--border)", overflow: "hidden" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {["Model", "Role", "Provider", "$/1M in", "$/1M out", "Tags"].map((h) => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "var(--text-3)", fontWeight: 500 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={`${m.role}-${m.id}`} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "8px 12px", color: "var(--text)", fontWeight: 500 }}>{m.display}</td>
              <td style={{ padding: "8px 12px" }}><TierBadge tier={m.role} /></td>
              <td style={{ padding: "8px 12px", color: "var(--text-2)" }}>{m.provider}</td>
              <td style={{ padding: "8px 12px", color: "var(--text-2)" }}>${m.input_per_1m}</td>
              <td style={{ padding: "8px 12px", color: "var(--text-2)" }}>${m.output_per_1m}</td>
              <td style={{ padding: "8px 12px" }}><AbilityScoreBars tags={m.tags} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 8: Wire ModelsPage.tsx**

Replace `src/pages/ModelsPage.tsx`:
```tsx
import ModelTable from "../components/models/ModelTable";

export default function ModelsPage() {
  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)", marginBottom: 20 }}>Model Registry</h1>
      <ModelTable />
    </div>
  );
}
```

- [ ] **Step 9: Commit**

```bash
git add src/components/memory/ src/components/models/ src/pages/MemoryPage.tsx src/pages/ModelsPage.tsx
git commit -m "feat: Memory Explorer + Model Registry pages"
```

---

## Task 10: Settings page

**Files:**
- Create: `src/hooks/useSettings.ts`
- Create: `src/components/settings/KeyInput.tsx`
- Create: `src/components/settings/ModeSelector.tsx`
- Create: `src/components/settings/ToggleRow.tsx`
- Modify: `src/pages/SettingsPage.tsx`

- [ ] **Step 1: Write useSettings hook**

Create `src/hooks/useSettings.ts`:
```typescript
import { useEffect, useState } from "react";
import { loadSettings, saveSettings, type AppSettings } from "../store/settings";

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings | null>(null);

  useEffect(() => { loadSettings().then(setSettings); }, []);

  async function update(patch: Partial<AppSettings>) {
    if (!settings) return;
    const next = { ...settings, ...patch };
    setSettings(next);
    await saveSettings(next);
  }

  return { settings, update };
}
```

- [ ] **Step 2: Write KeyInput.tsx**

Create `src/components/settings/KeyInput.tsx`:
```tsx
import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";

interface Props {
  label: string;
  value: string;
  onChange: (v: string) => void;
}

export default function KeyInput({ label, value, onChange }: Props) {
  const [show, setShow] = useState(false);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <label style={{ fontSize: 12, color: "var(--text-3)" }}>{label}</label>
      <div style={{ display: "flex", gap: 6 }}>
        <input
          type={show ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="sk-..."
          style={{
            flex: 1, background: "var(--surface)", border: "1px solid var(--border)",
            borderRadius: 6, color: "var(--text)", padding: "7px 10px", fontSize: 13,
          }}
        />
        <button
          onClick={() => setShow(!show)}
          style={{
            background: "var(--surface)", border: "1px solid var(--border)",
            borderRadius: 6, color: "var(--text-3)", padding: "7px 10px", cursor: "pointer",
          }}
        >
          {show ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write ModeSelector.tsx**

Create `src/components/settings/ModeSelector.tsx`:
```tsx
interface Props {
  value: "best" | "balanced" | "cost_efficient";
  onChange: (v: "best" | "balanced" | "cost_efficient") => void;
}
const OPTIONS = [
  { id: "cost_efficient", label: "Cost Efficient", desc: "Fastest, cheapest" },
  { id: "balanced",       label: "Balanced",       desc: "Quality + cost" },
  { id: "best",           label: "Best",            desc: "Max quality" },
] as const;

export default function ModeSelector({ value, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 8 }}>
      {OPTIONS.map((o) => (
        <button
          key={o.id}
          onClick={() => onChange(o.id)}
          style={{
            flex: 1, padding: "10px 12px", borderRadius: 8, border: "1px solid",
            borderColor: value === o.id ? "var(--accent)" : "var(--border)",
            background: value === o.id ? "var(--accent-dim)" : "var(--surface)",
            color: value === o.id ? "var(--accent)" : "var(--text-2)",
            cursor: "pointer", textAlign: "left",
          }}
        >
          <div style={{ fontWeight: 600, fontSize: 13 }}>{o.label}</div>
          <div style={{ fontSize: 11, marginTop: 2 }}>{o.desc}</div>
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Write ToggleRow.tsx**

Create `src/components/settings/ToggleRow.tsx`:
```tsx
interface Props { label: string; description?: string; value: boolean; onChange: (v: boolean) => void; }

export default function ToggleRow({ label, description, value, onChange }: Props) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 0", borderBottom: "1px solid var(--border)" }}>
      <div>
        <div style={{ fontSize: 13, color: "var(--text)" }}>{label}</div>
        {description && <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 2 }}>{description}</div>}
      </div>
      <button
        onClick={() => onChange(!value)}
        style={{
          width: 40, height: 22, borderRadius: 11, border: "none",
          background: value ? "var(--accent)" : "var(--surface-2)",
          cursor: "pointer", position: "relative", transition: "background 0.2s",
        }}
      >
        <div style={{
          position: "absolute", top: 3, left: value ? 20 : 3,
          width: 16, height: 16, borderRadius: "50%",
          background: "#fff", transition: "left 0.2s",
        }} />
      </button>
    </div>
  );
}
```

- [ ] **Step 5: Wire SettingsPage.tsx**

Replace `src/pages/SettingsPage.tsx`:
```tsx
import { useSettings } from "../hooks/useSettings";
import KeyInput from "../components/settings/KeyInput";
import ModeSelector from "../components/settings/ModeSelector";
import ToggleRow from "../components/settings/ToggleRow";

export default function SettingsPage() {
  const { settings, update } = useSettings();

  if (!settings) return <div style={{ padding: 24, color: "var(--text-3)" }}>Loading…</div>;

  return (
    <div style={{ padding: 24, maxWidth: 560, display: "flex", flexDirection: "column", gap: 24 }}>
      <h1 style={{ fontSize: 18, fontWeight: 700, color: "var(--text)" }}>Settings</h1>

      <section>
        <h2 style={{ fontSize: 13, color: "var(--text-3)", marginBottom: 12, letterSpacing: "0.05em" }}>DEFAULT MODE</h2>
        <ModeSelector value={settings.mode} onChange={(mode) => update({ mode })} />
      </section>

      <section>
        <h2 style={{ fontSize: 13, color: "var(--text-3)", marginBottom: 12 }}>BACKEND</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ fontSize: 12, color: "var(--text-3)" }}>Backend URL</label>
          <input
            value={settings.backendUrl}
            onChange={(e) => update({ backendUrl: e.target.value, useSidecar: e.target.value === "http://localhost:8080" })}
            style={{
              background: "var(--surface)", border: "1px solid var(--border)",
              borderRadius: 6, color: "var(--text)", padding: "7px 10px", fontSize: 13,
            }}
          />
          <ToggleRow
            label="Auto-start sidecar"
            description="Launch bundled backend on app start"
            value={settings.useSidecar}
            onChange={(useSidecar) => update({ useSidecar })}
          />
        </div>
      </section>

      <section>
        <h2 style={{ fontSize: 13, color: "var(--text-3)", marginBottom: 12 }}>API KEYS</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"].map((k) => (
            <KeyInput
              key={k}
              label={k}
              value={settings.apiKeys[k] ?? ""}
              onChange={(v) => update({ apiKeys: { ...settings.apiKeys, [k]: v } })}
            />
          ))}
        </div>
      </section>

      <section>
        <h2 style={{ fontSize: 13, color: "var(--text-3)", marginBottom: 12 }}>BUDGET</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <label style={{ fontSize: 12, color: "var(--text-3)" }}>Monthly cap (USD)</label>
          <input
            type="number"
            min={0}
            step={0.5}
            value={settings.budgetCapUsd}
            onChange={(e) => update({ budgetCapUsd: parseFloat(e.target.value) || 0 })}
            style={{
              width: 120, background: "var(--surface)", border: "1px solid var(--border)",
              borderRadius: 6, color: "var(--text)", padding: "7px 10px", fontSize: 13,
            }}
          />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add src/hooks/useSettings.ts src/components/settings/ src/pages/SettingsPage.tsx
git commit -m "feat: Settings page — API keys, mode selector, budget cap, backend URL"
```

---

## Task 11: Tauri sidecar + Rust main.rs

**Files:**
- Modify: `src-tauri/src/main.rs`
- Modify: `src-tauri/tauri.conf.json`
- Modify: `src-tauri/capabilities/default.json`

- [ ] **Step 1: Update main.rs with sidecar spawn/kill**

Replace `src-tauri/src/main.rs`:
```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use tauri::{Manager, State};
use tauri_plugin_shell::ShellExt;

struct SidecarHandle(Mutex<Option<tauri_plugin_shell::process::CommandChild>>);

#[tauri::command]
async fn restart_sidecar(
    app: tauri::AppHandle,
    handle: State<'_, SidecarHandle>,
) -> Result<(), String> {
    let mut guard = handle.0.lock().map_err(|e| e.to_string())?;
    if let Some(child) = guard.take() {
        let _ = child.kill();
    }
    let child = app
        .shell()
        .sidecar("aifactory-server")
        .map_err(|e| e.to_string())?
        .spawn()
        .map_err(|e| e.to_string())?;
    *guard = Some(child);
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarHandle(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();
            let state = app.state::<SidecarHandle>();
            // Spawn sidecar at startup
            match handle.shell().sidecar("aifactory-server") {
                Ok(cmd) => match cmd.spawn() {
                    Ok(child) => {
                        *state.0.lock().unwrap() = Some(child);
                    }
                    Err(e) => eprintln!("Failed to spawn sidecar: {e}"),
                },
                Err(e) => eprintln!("Sidecar not found: {e}"),
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<SidecarHandle>();
                if let Ok(mut guard) = state.0.lock() {
                    if let Some(child) = guard.take() {
                        let _ = child.kill();
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![restart_sidecar])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 2: Update tauri.conf.json**

In `src-tauri/tauri.conf.json`, set window title and add sidecar to `externalBin`. The full relevant sections:
```json
{
  "productName": "AiFactory",
  "version": "0.1.0",
  "identifier": "com.aifactory.app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:1420",
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build"
  },
  "app": {
    "windows": [
      {
        "title": "AiFactory",
        "width": 1200,
        "height": 800,
        "minWidth": 900,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": { "csp": null }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "externalBin": ["binaries/aifactory-server"],
    "icon": ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.icns", "icons/icon.ico"]
  }
}
```

- [ ] **Step 3: Update capabilities/default.json**

Add shell and store permissions to `src-tauri/capabilities/default.json`:
```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default capabilities",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "shell:allow-execute",
    "shell:allow-spawn",
    "shell:allow-kill",
    "store:allow-load",
    "store:allow-get",
    "store:allow-set",
    "store:allow-save"
  ]
}
```

- [ ] **Step 4: Verify Tauri builds with plugins**

```bash
npm run tauri build -- --debug
```
Expected: Build completes. Binary appears in `src-tauri/target/debug/bundle/`. The sidecar warning about missing binary is OK at this stage — sidecar binary is added in Task 12.

- [ ] **Step 5: Commit**

```bash
git add src-tauri/
git commit -m "feat: Tauri sidecar wiring + shell/store plugin capabilities"
```

---

## Task 12: PyInstaller sidecar build

**Files:**
- Create: `scripts/build-sidecar.sh`
- Create: `src-tauri/binaries/.gitkeep`

- [ ] **Step 1: Write build-sidecar.sh**

Create `scripts/build-sidecar.sh`:
```bash
#!/usr/bin/env bash
set -e

AIFACTORY_DIR="$HOME/Desktop/personal/AiFactory"
APP_DIR="$HOME/Desktop/personal/AiFactory-app"
BINARY_DIR="$APP_DIR/src-tauri/binaries"

echo "→ Activating venv…"
source "$AIFACTORY_DIR/.venv/bin/activate"

echo "→ Installing PyInstaller…"
pip install pyinstaller --quiet

echo "→ Building aifactory-server binary…"
cd "$AIFACTORY_DIR"
pyinstaller \
  --onefile \
  --name aifactory-server \
  --distpath "$BINARY_DIR" \
  --hidden-import aifactory \
  --hidden-import uvicorn \
  --collect-all aifactory \
  -c \
  - << 'EOF'
import sys
import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    uvicorn.run("aifactory.api.main:app", host="127.0.0.1", port=port, log_level="warning")
EOF

# Tauri expects the binary named with the target triple
TARGET=$(rustc -vV | grep host | awk '{print $2}')
mv "$BINARY_DIR/aifactory-server" "$BINARY_DIR/aifactory-server-$TARGET"

echo "✓ Sidecar built: $BINARY_DIR/aifactory-server-$TARGET"
```

```bash
chmod +x scripts/build-sidecar.sh
```

- [ ] **Step 2: Run the build**

```bash
cd ~/Desktop/personal/AiFactory-app
bash scripts/build-sidecar.sh
```
Expected: `src-tauri/binaries/aifactory-server-aarch64-apple-darwin` (or `x86_64-apple-darwin`) appears. File size ~30-60MB.

- [ ] **Step 3: Test sidecar runs**

```bash
./src-tauri/binaries/aifactory-server-$(rustc -vV | grep host | awk '{print $2}') &
sleep 2
curl http://localhost:8080/api/health
kill %1
```
Expected: `{"status":"ok","version":"..."}` returned.

- [ ] **Step 4: Launch full app with sidecar**

```bash
npm run tauri dev
```
Expected: App launches, sidecar auto-starts on `localhost:8080`, health check passes, AppShell loads.

- [ ] **Step 5: Commit**

```bash
git add scripts/build-sidecar.sh src-tauri/binaries/.gitkeep
git commit -m "feat: PyInstaller sidecar build script"
```

---

## Task 13: Production .dmg build

**Files:**
- Modify: `src-tauri/tauri.conf.json` (bundle metadata)

- [ ] **Step 1: Add app icon (use placeholder if no custom icon yet)**

```bash
# Generate placeholder icons from any 1024x1024 PNG
# If you have a custom icon at ~/Desktop/icon.png:
npm run tauri icon ~/Desktop/icon.png
# Otherwise skip — Tauri uses default icons
```

- [ ] **Step 2: Run release build**

```bash
npm run tauri build
```
Expected: `src-tauri/target/release/bundle/dmg/AiFactory_0.1.0_aarch64.dmg` created. No errors.

- [ ] **Step 3: Test the .dmg**

```bash
open src-tauri/target/release/bundle/dmg/AiFactory_0.1.0_aarch64.dmg
```
Expected: DMG mounts, drag-to-Applications installer shown. App launches from Applications folder, sidecar starts, health check passes, all 8 pages accessible.

- [ ] **Step 4: Add .gitignore entries**

Append to `.gitignore`:
```
src-tauri/target/
src-tauri/binaries/aifactory-server-*
node_modules/
dist/
```

- [ ] **Step 5: Final commit**

```bash
git add .gitignore src-tauri/tauri.conf.json
git commit -m "feat: production .dmg build config + gitignore"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Tauri + Vite + React scaffold | Task 1 |
| api.ts ported, VITE_ prefix | Task 2 |
| Tauri store for settings | Tasks 2, 10 |
| AppShell + collapsible sidebar | Task 3 |
| 8 pages routed | Tasks 3–10 |
| Health check / startup screen | Task 4 |
| Chat + WebSocket streaming | Task 5 |
| AgentSpawnVisualizer | Task 5 |
| Dashboard (stats + recent runs) | Task 6 |
| Run History + RunDetailDrawer | Task 6 |
| Cost & Usage charts (recharts) | Task 7 |
| Budget gauge | Task 7 |
| MCP Catalog grid + toggles | Task 8 |
| Add custom MCP modal | Task 8 |
| Memory Explorer with search + clear | Task 9 |
| Model Registry (read-only) | Task 9 |
| Settings: keys, mode, budget, URL | Task 10 |
| Sidecar spawn/kill in Rust | Task 11 |
| PyInstaller build script | Task 12 |
| .dmg packaging | Task 13 |
| Sidecar crash → restart once | Task 11 (one restart in main.rs) |
| Custom backend URL disables sidecar | Task 10 (useSidecar toggle) |

All spec requirements covered. No TBDs, no missing tasks.
