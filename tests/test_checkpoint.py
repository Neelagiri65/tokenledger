"""
Tests for the checkpoint log + living dashboard (tokenledger/checkpoint.py, explainer.py).
The checkpoint log is the durable progress record that survives a limit and drives the timeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.checkpoint import (  # noqa: E402
    Checkpoint, add_checkpoint, load_checkpoints, timeline_html,
)
from tokenledger.explainer import write_dashboard  # noqa: E402
from tokenledger.manifest import Run, record_run  # noqa: E402

CP = "_t_checkpoints.jsonl"
RUN = "_t_runs.jsonl"
OUT = "_t_dashboard.html"


def _clean():
    for f in (CP, RUN, OUT):
        if os.path.exists(f):
            os.remove(f)


def test_checkpoint_roundtrip_and_order():
    _clean()
    add_checkpoint(Checkpoint("2026-06-22T19:00:00Z", "first", "did a thing", "abc123", "build"), CP)
    add_checkpoint(Checkpoint("2026-06-22T20:00:00Z", "second", phase="resilience"), CP)
    cps = load_checkpoints(CP)
    assert [c.title for c in cps] == ["first", "second"]          # append order preserved
    assert cps[0].commit == "abc123" and cps[1].detail == ""
    os.remove(CP)


def test_timeline_html_renders_and_escapes():
    _clean()
    add_checkpoint(Checkpoint("2026-06-22T19:00:00Z", "wedge <proven>", "we aren't cooperative", "abc123"), CP)
    h = timeline_html(CP)
    assert "<ol class='timeline'>" in h and "abc123" in h
    assert "&lt;proven&gt;" in h and "aren&#x27;t" in h           # HTML-escaped, no injection
    assert timeline_html("_does_not_exist.jsonl").startswith("<p")  # graceful empty
    os.remove(CP)


def test_living_dashboard_injects_timeline_and_runs():
    _clean()
    add_checkpoint(Checkpoint("2026-06-22T19:00:00Z", "milestone-X", "detail-Y", "deadbee"), CP)
    record_run(Run("tA", "wf_a", "job-Z", status="running", phase="Build"), RUN)
    out = write_dashboard(OUT, generated_at="2026-06-22T22:00:00Z", run_path=RUN, checkpoint_path=CP)
    html_text = open(out, encoding="utf-8").read()
    assert "milestone-X" in html_text and "detail-Y" in html_text   # timeline injected
    assert "job-Z" in html_text and "● running" in html_text        # run snapshot injected
    assert "2026-06-22T22:00:00Z" in html_text                      # generation stamp
    assert "%%TIMELINE%%" not in html_text and "%%RUNS%%" not in html_text  # all placeholders filled
    _clean()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all checkpoint tests passed")
