"""
Run manifest — the durable record of every workflow run, so work SURVIVES a surprise session or
context limit (see docs/architecture-cockpit.md). It is the resilience spine: a session death loses
the live `/workflows` view, never the manifest.

Critically, each record stores the run's FULL `args`, so a resume can NEVER drop them — the exact
failure hit this session, when a bare `scriptPath` resume lost its research question and errored
instantly. With the manifest, resume is mechanical: read it, re-invoke incomplete runs WITH their
recorded args.

One JSON object per line (JSONL). Append on launch; update on phase change / completion. Pure
stdlib, no network (NO EGRESS). No clock in here — the caller supplies `started` (mirrors core.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

DEFAULT_PATH = "run-manifest.jsonl"

# terminal statuses — a run in any of these is not resumable / not in-flight
_TERMINAL = {"completed", "stopped"}


@dataclass
class Run:
    task_id: str                 # the background task id (e.g. w83goecq6)
    run_id: str                  # the workflow run id (e.g. wf_efc588c1-df2) — needed to resume
    name: str                    # short workflow name
    status: str = "running"      # running | completed | stopped | failed
    phase: str = ""              # last known phase (Spec/Build/Review/Gate/...)
    args: str = ""               # FULL args — required for a correct resume; never truncate at rest
    script_path: str = ""        # persisted workflow script path — needed to resume
    started: str = ""            # ISO timestamp, supplied by the caller
    note: str = ""


def record_run(run: Run, path: str = DEFAULT_PATH) -> None:
    """Append a run on launch."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")


def load_runs(path: str = DEFAULT_PATH) -> list[Run]:
    if not os.path.exists(path):
        return []
    out: list[Run] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Run(**json.loads(line)))
    return out


def update_run(task_id: str, path: str = DEFAULT_PATH, **fields: object) -> bool:
    """Update the latest record for a task_id (phase/status/note). Returns True if one was found."""
    runs = load_runs(path)
    found = False
    for r in runs:
        if r.task_id == task_id:
            for k, v in fields.items():
                if hasattr(r, k):
                    setattr(r, k, v)
            found = True
    if found:
        _rewrite(runs, path)
    return found


def incomplete(path: str = DEFAULT_PATH) -> list[Run]:
    """Runs that are not in a terminal state — i.e. candidates for resume after a limit-kill."""
    return [r for r in load_runs(path) if r.status not in _TERMINAL]


def resume_command(run: Run) -> str:
    """The exact, copy-pasteable resume invocation — WITH full args, so it never drops them.

    This is the antidote to the bug observed this session. The args are emitted in full (not
    truncated) precisely because a resume that omits them fails.
    """
    if not run.script_path or not run.run_id:
        return "(not resumable: missing script_path / run_id)"
    base = f'Workflow({{scriptPath: "{run.script_path}", resumeFromRunId: "{run.run_id}"'
    if run.args:
        args = run.args.replace("\\", "\\\\").replace('"', '\\"')
        return base + f', args: "{args}"}})'
    return base + "})"


def _rewrite(runs: list[Run], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
