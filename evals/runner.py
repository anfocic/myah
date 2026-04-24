"""Eval runner: load task modules, invoke `run_agent`, grade, log.

Shape, per CONCEPTS-style pedagogy:

    for task in tasks:
        setup() → chdir into tmp fixture if needed
        swap provider if task pins one
        capture trace via on_tool_start/on_tool_end
        run_agent(...) with a thread + wall_timeout_s
        grade each check against a bundle of (content, trace, cwd, ...)
        append one JSONL line; print summary row

Everything the runner does is layered on top of `run_agent`'s existing
public surface — no changes required to the agent loop. The precedent
for calling `run_agent` as a library is `tools/subagent.py:spawn_subagent`.
"""
from __future__ import annotations

import importlib
import json
import os
import pkgutil
import shutil
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agent import READ_ONLY_TOOLS
from agent.loop import run_agent
from evals import checks as checks_mod
from providers import build_provider, get_active_provider, set_active_provider
from repl.state import new_state
from repl.tool_registry import make_execute_tool
from repl.tool_registry import tools as ALL_TOOLS

EVALS_ROOT = Path(__file__).parent
FIXTURES_ROOT = EVALS_ROOT / "fixtures"
RESULTS_ROOT = EVALS_ROOT / "results"


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    content: str
    trace: list[dict]
    check_results: list[dict]
    # ctx_used is the provider's reported prompt-token count when surfaced;
    # otherwise the loop's char/4 fallback estimate. Not disentangled here.
    ctx_used: int
    completion_tokens: int | None
    wall_s: float
    timeout: bool
    over_tool_budget: bool
    provider: str
    model: str
    error: str | None = None


def discover_tasks() -> list[dict]:
    """Walk `evals/tasks/` and import every module that exports `TASK`."""
    found = []
    pkg_path = EVALS_ROOT / "tasks"
    for info in pkgutil.iter_modules([str(pkg_path)]):
        if info.ispkg or info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"evals.tasks.{info.name}")
        task = getattr(mod, "TASK", None)
        if task is None:
            continue
        found.append(task)
    return found


def _permission_factory(mode: str):
    if mode == "allow_all":
        return lambda name, args, meta=None: True
    if mode == "deny_all":
        return lambda name, args, meta=None: False
    if mode == "readonly_only":
        return lambda name, args, meta=None: name in READ_ONLY_TOOLS
    raise ValueError(f"unknown permission mode: {mode!r}")


def _resolve_provider(task: dict, cli_provider: str | None, cli_model: str | None):
    """Precedence: CLI > task > current active. Returns (name, model) and
    sets the active provider as a side effect if a swap is needed."""
    task_pin = task.get("provider")  # tuple[str,str] or None
    task_model = task_pin[1] if task_pin else None
    if cli_provider:
        model = cli_model or task_model or get_active_provider().model
        set_active_provider(build_provider(cli_provider, model))
        return cli_provider, model
    if task_pin:
        name, model = task_pin
        set_active_provider(build_provider(name, model))
        return name, model
    active = get_active_provider()
    return active.name, active.model


def _copy_fixture(fixture_name: str) -> tuple[Path, Path]:
    src = FIXTURES_ROOT / fixture_name
    if not src.is_dir():
        raise FileNotFoundError(f"fixture not found: {src}")
    tmp = Path(tempfile.mkdtemp(prefix=f"mia-eval-{fixture_name}-"))
    # dirs_exist_ok: copytree into a freshly created tmp dir that already
    # exists because mkdtemp made it.
    shutil.copytree(src, tmp, dirs_exist_ok=True)
    return tmp, src


def _build_trace_callbacks() -> tuple[Any, Any, Any]:
    """Capture tool-call traces via the loop's existing callback hooks.

    The loop calls `on_tool_start(name, args)` strictly in request order
    (serial for loop in `_run_tools_parallel`) and `on_tool_end(name, args,
    result, ok)` in completion order. We match end → start by consuming
    the oldest unmatched entry whose name equals the end's name, which
    handles parallel dispatch correctly for any realistic case (the only
    ambiguity is two in-flight calls to the same tool, which still end
    up counted; only the result/ok pairing can shuffle, which doesn't
    affect grading)."""
    entries: list[dict] = []
    lock = threading.Lock()

    def on_start(name, args):
        with lock:
            entries.append({
                "name": name,
                "args": args,
                "ok": None,
                "result_head": "",
                "_matched": False,
            })

    def on_end(name, args, result, ok):
        with lock:
            for entry in entries:
                if entry["_matched"] or entry["name"] != name:
                    continue
                entry["ok"] = ok
                head = str(result or "")
                entry["result_head"] = head if len(head) <= 200 else head[:197] + "..."
                entry["_matched"] = True
                return

    def snapshot() -> list[dict]:
        with lock:
            return [{k: v for k, v in e.items() if k != "_matched"} for e in entries]

    return on_start, on_end, snapshot


def _run_one(task: dict, cli_provider: str | None, cli_model: str | None) -> TaskResult:
    task_id = task["id"]
    limits = task.get("limits", {})
    max_tool_calls = limits.get("max_tool_calls", 20)
    wall_timeout_s = limits.get("wall_timeout_s", 120)

    setup = task.get("setup") or {}
    fs_fixture = setup.get("fs")
    fixture_dir: Path | None = None
    tmp_cwd: Path | None = None
    original_cwd = Path(os.getcwd())
    cwd_for_task = original_cwd

    if fs_fixture:
        tmp_cwd, fixture_dir = _copy_fixture(fs_fixture)
        os.chdir(tmp_cwd)
        cwd_for_task = tmp_cwd

    # Shared across try/finally so the cleanup path can decide whether
    # it's safe to rmtree the tempdir (not safe while the thread is live).
    thread_still_alive = [False]

    try:
        provider_name, model_name = _resolve_provider(task, cli_provider, cli_model)
        permission_check = _permission_factory(task.get("permission", "allow_all"))
        on_start, on_end, snapshot = _build_trace_callbacks()

        result_box: dict[str, Any] = {}

        def _invoke():
            try:
                content, _hist, ctx_used, stats = run_agent(
                    user_input=task["prompt"],
                    tools=ALL_TOOLS,
                    execute_tool=make_execute_tool(new_state(), permission_check),
                    history=[],
                    console=None,
                    permission_check=permission_check,
                    plan_mode=task.get("plan_mode", False),
                    on_tool_start=on_start,
                    on_tool_end=on_end,
                )
                result_box["content"] = content
                result_box["ctx_used"] = ctx_used
                result_box["stats"] = stats or {}
            except Exception as e:
                result_box["error"] = f"{type(e).__name__}: {e}"
                result_box["traceback"] = traceback.format_exc()

        t0 = time.monotonic()
        thread = threading.Thread(target=_invoke, daemon=True)
        thread.start()
        thread.join(wall_timeout_s)
        wall_s = time.monotonic() - t0
        timed_out = thread.is_alive()
        thread_still_alive[0] = timed_out

        trace = snapshot()
        over_budget = len(trace) > max_tool_calls

        content = result_box.get("content", "")
        stats = result_box.get("stats", {}) or {}
        ctx_used = result_box.get("ctx_used", 0)
        error = result_box.get("error")

        bundle = {
            "content": content,
            "trace": trace,
            "stats": stats,
            "ctx_used": ctx_used,
            "cwd": cwd_for_task,
            "fixture_dir": fixture_dir,
        }

        check_results = []
        task_passed = not timed_out and not over_budget and not error
        for check in task.get("checks", []):
            ok, why = checks_mod.dispatch(check, bundle)
            kind = check.get("type") if isinstance(check, dict) else "callable"
            check_results.append({"type": kind, "pass": ok, "why": why})
            if not ok:
                task_passed = False

        if timed_out and error is None:
            error = f"wall_timeout_s={wall_timeout_s} exceeded"

        return TaskResult(
            task_id=task_id,
            passed=task_passed,
            content=content,
            trace=trace,
            check_results=check_results,
            ctx_used=ctx_used,
            completion_tokens=stats.get("completion_tokens"),
            wall_s=wall_s,
            timeout=timed_out,
            over_tool_budget=over_budget,
            provider=provider_name,
            model=model_name,
            error=error,
        )
    finally:
        # Always restore cwd — subsequent tasks need a known root. Skip
        # rmtree if the thread timed out: the still-running `run_agent` may
        # still be mutating files inside tmp_cwd, and deleting them under it
        # would raise or produce incoherent tool errors. Accept the leaked
        # temp dir — the v1 scope cut on thread cancellation already
        # acknowledges this (see CONCEPTS §45).
        if tmp_cwd is not None:
            os.chdir(original_cwd)
            if not thread_still_alive[0]:
                shutil.rmtree(tmp_cwd, ignore_errors=True)


def _write_results_jsonl(results: list[TaskResult]) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not results:
        return RESULTS_ROOT / "empty.jsonl"
    first = results[0]
    ts = int(time.time())
    safe_model = first.model.replace("/", "_").replace(":", "_")
    path = RESULTS_ROOT / f"{first.provider}-{safe_model}-{ts}.jsonl"
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "task": r.task_id,
                "provider": r.provider,
                "model": r.model,
                "pass": r.passed,
                "checks": r.check_results,
                "tool_calls": len(r.trace),
                "ctx_used": r.ctx_used,
                "completion_tokens": r.completion_tokens,
                "wall_s": round(r.wall_s, 3),
                "timeout": r.timeout,
                "over_tool_budget": r.over_tool_budget,
                "error": r.error,
                "trace": r.trace,
                "content": r.content,
            }) + "\n")
    return path


def _print_summary(results: list[TaskResult], console: Console, jsonl_path: Path) -> None:
    table = Table(title="Eval results")
    table.add_column("task")
    table.add_column("pass")
    table.add_column("tools")
    table.add_column("ctx")
    table.add_column("gen tok")
    table.add_column("wall (s)")
    table.add_column("notes")
    for r in results:
        notes = []
        if r.timeout:
            notes.append("timeout")
        if r.over_tool_budget:
            notes.append("over-budget")
        if r.error and not r.timeout:
            notes.append("error")
        for c in r.check_results:
            if not c["pass"]:
                notes.append(f"{c['type']}!")
        table.add_row(
            r.task_id,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            str(len(r.trace)),
            str(r.ctx_used or "-"),
            str(r.completion_tokens or "-"),
            f"{r.wall_s:.2f}",
            ", ".join(notes) or "-",
        )
    n_pass = sum(1 for r in results if r.passed)
    table.add_section()
    table.add_row(
        "[bold]total[/bold]",
        f"{n_pass}/{len(results)}",
        "", "", "", "",
        f"→ {jsonl_path.name}",
    )
    console.print(table)


def run_suite(
    task_ids: list[str] | None = None,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    console: Console | None = None,
) -> list[TaskResult]:
    console = console or Console()
    all_tasks = discover_tasks()
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in all_tasks if t["id"] in wanted]
        missing = wanted - {t["id"] for t in tasks}
        if missing:
            console.print(f"[yellow]unknown task ids: {sorted(missing)}[/yellow]")
    else:
        tasks = all_tasks

    # Save/restore the active provider so calling `run_suite` in-process
    # (tests, a future /eval slash command) doesn't leak the last task's
    # provider into the surrounding REPL session.
    saved_provider = get_active_provider()
    results: list[TaskResult] = []
    try:
        for task in tasks:
            console.print(f"[dim]→ running {task['id']}...[/dim]")
            r = _run_one(task, cli_provider, cli_model)
            results.append(r)
    finally:
        set_active_provider(saved_provider)

    if results:
        jsonl_path = _write_results_jsonl(results)
    else:
        jsonl_path = RESULTS_ROOT / "empty.jsonl"
    _print_summary(results, console, jsonl_path)
    return results


def list_tasks() -> list[str]:
    return [t["id"] for t in discover_tasks()]
