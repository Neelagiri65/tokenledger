"""
Tests for the run manifest (tokenledger/manifest.py) — the resilience spine.

The load-bearing guarantee: a resume command for an incomplete run ALWAYS carries the run's full
args, so the args-drop failure observed this session can never recur. Also covers record/load,
update, incomplete filtering, and well-formed (balanced-brace) resume commands.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenledger.manifest import (  # noqa: E402
    Run, record_run, load_runs, update_run, incomplete, resume_command,
)

DB = "_t_manifest.jsonl"


def _fresh():
    if os.path.exists(DB):
        os.remove(DB)


def test_record_and_load_roundtrip():
    _fresh()
    record_run(Run("t1", "wf_a", "alpha", status="running", phase="Build", started="2026-06-22T00:00:00Z"), DB)
    record_run(Run("t2", "wf_b", "beta", status="completed", phase="Gate"), DB)
    runs = load_runs(DB)
    assert [r.task_id for r in runs] == ["t1", "t2"]
    assert runs[0].phase == "Build" and runs[1].status == "completed"
    os.remove(DB)


def test_update_run():
    _fresh()
    record_run(Run("t1", "wf_a", "alpha", status="running", phase="Build"), DB)
    assert update_run("t1", DB, status="completed", phase="Gate")
    assert not update_run("nope", DB, status="completed")
    r = load_runs(DB)[0]
    assert r.status == "completed" and r.phase == "Gate"
    os.remove(DB)


def test_incomplete_filters_terminal():
    _fresh()
    for tid, st in [("a", "running"), ("b", "completed"), ("c", "stopped"), ("d", "failed")]:
        record_run(Run(tid, f"wf_{tid}", tid, status=st), DB)
    inc = {r.task_id for r in incomplete(DB)}
    assert inc == {"a", "d"}  # running + failed are resumable; completed/stopped are terminal
    os.remove(DB)


def test_resume_command_always_includes_full_args():
    # The regression guard for THE bug: a resume must carry the full args, never drop them.
    q = "Research the full landscape around TokenLedger — a no-egress metering tool. " * 4
    r = Run("t", "wf_x", "deep-research", status="running",
            script_path="/path/to/script.js", args=q)
    cmd = resume_command(r)
    assert "resumeFromRunId" in cmd and "wf_x" in cmd
    assert "args:" in cmd
    assert "no-egress metering tool" in cmd          # the FULL question is present, not truncated
    assert cmd.endswith('"})')                        # balanced braces, well-formed call


def test_resume_command_no_args_is_well_formed():
    r = Run("t", "wf_y", "sdd-loop", status="running", script_path="/path/s.js")
    cmd = resume_command(r)
    assert cmd == 'Workflow({scriptPath: "/path/s.js", resumeFromRunId: "wf_y"})'  # no stray brace


def test_resume_command_not_resumable_without_ids():
    assert "not resumable" in resume_command(Run("t", "", "x", script_path=""))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all manifest tests passed")
