"""
CLI commands for the Meta-Agent — system health reviewer.

  aifactory meta review [--last N] [--output PATH]   Review recent runs + produce recommendations
  aifactory meta calibrate [--apply] [--min-samples N]  Show / apply RL ability score patches
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(name="meta", help="🔬 Meta-agent: system self-improvement tools.")
console = Console()


@app.command("review")
def review(
    last: Annotated[int, typer.Option("--last", help="Number of recent runs to analyse")] = 20,
    output: Annotated[
        Path | None, typer.Option("--output", help="Write report to this file")
    ] = None,
    model: Annotated[str, typer.Option("--model", help="LLM to use for analysis")] = "",
) -> None:
    """
    Run the meta-agent to review recent pipeline runs and produce
    system improvement recommendations.
    """

    async def _run() -> str:
        from aifactory.core.model_selector import load_agent_cfg
        from aifactory.core.rl_scorer import get_model_rl_scores
        from aifactory.core.run_store import get_run_store
        from aifactory.graphs.dynamic_pipeline import PipelineAgent

        console.print(f"[dim]Fetching last {last} runs from run store...[/dim]")
        store = get_run_store()
        records = store.list_recent(n=last)

        if not records:
            return "No run history found. Run at least one pipeline first."

        # Build summary statistics
        total = len(records)
        flagged = sum(r.quality_flagged_count > 0 for r in records)
        avg_quality = sum(r.composite_quality for r in records) / total
        mode_counts: dict[str, int] = {}
        gate_counts: dict[str, int] = {}
        model_quality: dict[str, list[float]] = {}

        for rec in records:
            mode_counts[rec.mode] = mode_counts.get(rec.mode, 0) + 1
            gate_counts[rec.gate or "full"] = gate_counts.get(rec.gate or "full", 0) + 1
            for _role, m in rec.model_selections.items():
                model_quality.setdefault(m, []).append(rec.composite_quality)

        model_avg = {m: sum(qs) / len(qs) for m, qs in model_quality.items()}

        # Gather RL scores
        rl_data: dict[str, dict[str, float]] = {}
        for task_type in ["coding", "math", "research", "writing", "analysis", "general"]:
            try:
                scores = await get_model_rl_scores(task_type, min_samples=1)
                for model_id, score in scores.items():
                    rl_data.setdefault(model_id, {})[task_type] = score
            except Exception:
                pass

        # Build structured context for meta-agent
        context = {
            "runs_analysed": total,
            "runs_with_quality_flags": flagged,
            "flagged_rate_pct": round(100 * flagged / total, 1),
            "avg_composite_quality": round(avg_quality, 2),
            "mode_distribution": mode_counts,
            "gate_distribution": gate_counts,
            "model_avg_quality": {
                m: round(v, 2) for m, v in sorted(model_avg.items(), key=lambda x: -x[1])
            },
            "rl_scores_by_model_task": rl_data,
            "recent_runs": [
                {
                    "run_id": r.run_id[:16],
                    "query": r.query[:100],
                    "quality": r.composite_quality,
                    "flagged": r.quality_flagged_count,
                    "gate": r.gate or "full",
                    "task_types": r.task_types,
                }
                for r in records[:10]
            ],
        }

        console.print("[dim]Calling meta-agent LLM...[/dim]")

        # Load meta agent config
        try:
            cfg = load_agent_cfg("meta")
        except Exception:
            cfg = {
                "system_prompt": "You are a system analyst. Review the data and produce recommendations.",  # noqa: E501
                "max_tokens": 2000,
            }

        _model = model or "gpt-4o"
        agent = PipelineAgent("meta", {**cfg, "model": _model}, _model)

        report = await agent.think(
            f"Here is the system performance data for the last {total} runs:\n\n"
            f"```json\n{json.dumps(context, indent=2)}\n```\n\n"
            "Produce your analysis and recommendations.",
            {},
        )
        return report

    report = asyncio.run(_run())

    console.print(Panel(report, title="[bold cyan]Meta-Agent Report[/bold cyan]", expand=False))

    if output:
        output.write_text(report)
        console.print(f"[green]Report saved to {output}[/green]")


@app.command("calibrate")
def calibrate(
    apply: Annotated[
        bool, typer.Option("--apply", help="Write patches to model_registry.yaml")
    ] = False,
    min_samples: Annotated[int, typer.Option("--min-samples", help="Min RL samples required")] = 5,
    no_backup: Annotated[bool, typer.Option("--no-backup", help="Skip .bak copy")] = False,
) -> None:
    """
    Compute RL-driven ability score patches for model_registry.yaml.

    Without --apply, shows what would change (dry run).
    With --apply, writes patches with automatic backup.
    """

    async def _run() -> dict:
        from aifactory.core.model_calibrator import (
            apply_patches,
            get_calibration_report,
        )

        report = await get_calibration_report(min_samples=min_samples)
        console.print(Panel(report["summary"], title="[bold]Calibration Report[/bold]"))

        if not report["patches"]:
            return report

        if apply:
            result = apply_patches(
                report["patches"],
                dry_run=False,
                backup=not no_backup,
            )
            console.print(
                f"[green]Applied {result['applied']} model patches to {result['path']}[/green]"
            )
            if not no_backup:
                console.print(f"[dim]Backup: {result['path']}.bak[/dim]")
        else:
            console.print(
                "[yellow]Dry run — use --apply to write patches to model_registry.yaml[/yellow]"
            )

        return report

    asyncio.run(_run())
