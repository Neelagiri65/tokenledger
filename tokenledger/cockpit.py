"""
Cockpit — a READ-VIEW over the durable state spine (run manifest + git log). NOT a second system:
it only reflects last-known durable state, so a context reset or session limit loses nothing — the
cockpit simply shows where things stood (see docs/architecture-cockpit.md).

Two things matter operationally:
  1. Status of every run (phase, terminal or in-flight).
  2. RESUMABLE runs — incomplete runs with a ready-to-paste resume command (the resilience payoff).

Pure stdlib + a local `git log` shell-out. No network (NO EGRESS).
"""

from __future__ import annotations

import html
import subprocess

from .manifest import DEFAULT_PATH, Run, incomplete, load_runs, resume_command


def _git_log(n: int = 8) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "log", "--oneline", "-n", str(n)],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()
    except Exception:
        return []


def print_cockpit(path: str = DEFAULT_PATH) -> None:
    runs = load_runs(path)
    inflight = sum(1 for r in runs if r.status == "running")
    print(f"\nTokenLedger cockpit — {len(runs)} run(s), {inflight} in flight\n")
    print(f"{'task':<12}{'status':<11}{'phase':<14}{'name'}")
    print("-" * 72)
    for r in runs:
        print(f"{r.task_id:<12}{r.status:<11}{r.phase:<14}{r.name[:34]}")

    inc = incomplete(path)
    if inc:
        print(f"\n{len(inc)} resumable run(s) — paste to continue after a limit/reset:")
        for r in inc:
            print(f"  [{r.task_id}] {r.name}")
            print(f"    {resume_command(r)}")

    log = _git_log()
    if log:
        print("\nrecent commits (durable progress):")
        for line in log:
            print(f"  {line}")
    print()


def write_cockpit_html(path: str = DEFAULT_PATH, out: str = "cockpit.html") -> str:
    runs = load_runs(path)
    inc = incomplete(path)

    def status_cls(s: str) -> str:
        return {"running": "run", "failed": "fail", "stopped": "stop"}.get(s, "")

    rows = "".join(
        f"<tr class='{status_cls(r.status)}'><td>{html.escape(r.task_id)}</td>"
        f"<td>{html.escape(r.status)}</td><td>{html.escape(r.phase)}</td>"
        f"<td>{html.escape(r.name)}</td></tr>"
        for r in runs
    )
    resumable = "".join(
        f"<li><code>{html.escape(r.task_id)}</code> — {html.escape(r.name)}"
        f"<pre>{html.escape(resume_command(r))}</pre></li>"
        for r in inc
    ) or "<li>none — all runs terminal</li>"
    commits = "".join(f"<li><code>{html.escape(c)}</code></li>" for c in _git_log())

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>TokenLedger cockpit</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;color:#111;background:#fafafa}}
 h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin-top:1.6rem;color:#333}}
 table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 th,td{{padding:.45rem .7rem;text-align:left;border-bottom:1px solid #eee}}
 tr.run td{{background:#eef6ff}} tr.fail td{{background:#fff4f4}} tr.stop td{{background:#f6f6f6;color:#777}}
 code{{background:#f3f3f3;padding:.05rem .3rem;border-radius:3px}}
 pre{{background:#f3f3f3;padding:.5rem;border-radius:4px;white-space:pre-wrap;word-break:break-all;font-size:.8rem}}
 .note{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>TokenLedger cockpit</h1>
<p class="note">Read-view over the durable state spine (run manifest + git). Reflects last-known
state — survives a session limit or context reset.</p>
<h2>Runs</h2><table><tr><th>task</th><th>status</th><th>phase</th><th>name</th></tr>{rows}</table>
<h2>Resumable (paste to continue — args included, never dropped)</h2><ul>{resumable}</ul>
<h2>Recent commits (durable progress)</h2><ul>{commits}</ul>
</body></html>"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return out
