"""
Checkpoint log — a durable, append-only record of milestones, so progress is never lost to a
session/context limit AND can be shown as a timeline (see docs/architecture-cockpit.md). It is the
twin of the run manifest: the manifest tracks live workflow runs; the checkpoint log tracks
completed, human-meaningful progress.

One JSON object per line (JSONL). Append-only. Pure stdlib, no network (NO EGRESS). No clock here —
the caller supplies `ts` (mirrors core.py / manifest.py); the CLI is the edge that stamps time.
"""

from __future__ import annotations

import html
import json
import os
from dataclasses import asdict, dataclass

DEFAULT_PATH = "checkpoints.jsonl"


@dataclass
class Checkpoint:
    ts: str               # ISO timestamp, supplied by the caller
    title: str            # short milestone headline
    detail: str = ""      # one-line plain-language description
    commit: str = ""      # short git sha, if this milestone was committed
    phase: str = ""        # optional grouping: research | framework | build | resilience | ...


def add_checkpoint(cp: Checkpoint, path: str = DEFAULT_PATH) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(cp), ensure_ascii=False) + "\n")


def load_checkpoints(path: str = DEFAULT_PATH) -> list[Checkpoint]:
    if not os.path.exists(path):
        return []
    out: list[Checkpoint] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Checkpoint(**json.loads(line)))
    return out


def print_timeline(path: str = DEFAULT_PATH) -> None:
    cps = load_checkpoints(path)
    if not cps:
        print("(no checkpoints yet)")
        return
    print(f"\nProgress — {len(cps)} checkpoint(s):")
    for i, c in enumerate(cps, 1):
        sha = f" [{c.commit}]" if c.commit else ""
        when = (c.ts or "")[:16].replace("T", " ")
        print(f"  {i:>2}. {when}  {c.title}{sha}")
        if c.detail:
            print(f"      {c.detail}")
    print()


def timeline_html(path: str = DEFAULT_PATH) -> str:
    """Return an HTML <ol> timeline snippet for embedding in the living dashboard."""
    cps = load_checkpoints(path)
    if not cps:
        return "<p class='lead'>No checkpoints recorded yet.</p>"
    items = []
    for c in cps:
        when = html.escape((c.ts or "")[:16].replace("T", " "))
        sha = f"<code>{html.escape(c.commit)}</code>" if c.commit else ""
        phase = f"<span class='tag t-amber'>{html.escape(c.phase)}</span>" if c.phase else ""
        items.append(
            f"<li><span class='when'>{when}</span>"
            f"<b>{html.escape(c.title)}</b> {phase} {sha}"
            f"<span class='det'>{html.escape(c.detail)}</span></li>"
        )
    return "<ol class='timeline'>" + "".join(items) + "</ol>"
