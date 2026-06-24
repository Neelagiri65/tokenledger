"""
TokenLedger CLI — the outsider-runnable entry point.

  retoken demo                         # offline demo (no keys), writes db + html + report
  retoken ingest <file> [--format litellm|jsonl] [--db PATH]
  retoken report [--db PATH] [--html PATH] [--md PATH]

Quickstart (sidecar over an existing LiteLLM gateway):
  retoken ingest litellm_spendlogs.jsonl --format litellm --db retoken.db
  retoken report --db retoken.db --html report.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .core import CallRecord, Usage
from .store import Store
from .dashboard import print_summary, write_html, discrepancy_report_md
from .connectors import ingest_litellm_spendlog


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import main as _demo_main
    _demo_main()
    return 0


def _opt_float(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None  # reject inf/nan: can't poison a money sum


def _ingest_jsonl_native(path: str, store: Store) -> int:
    """Ingest TokenLedger-native event JSONL: {provider,model,route,user_id,session_id,ts,
    reported:{input_tokens,...}, request_text, response_text}."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rep = d.get("reported", {})
            store.record(CallRecord(
                provider=d.get("provider", "openai"), model=d["model"],
                route=d.get("route", "/chat/completions"),
                user_id=str(d.get("user_id", "unknown")),
                session_id=str(d.get("session_id", "unknown")),
                ts=str(d.get("ts", "")),
                reported=Usage(
                    input_tokens=int(rep.get("input_tokens", 0)),
                    output_tokens=int(rep.get("output_tokens", 0)),
                    reasoning_tokens=int(rep.get("reasoning_tokens", 0)),
                    cache_read_tokens=int(rep.get("cache_read_tokens", 0)),
                    cache_creation_tokens=int(rep.get("cache_creation_tokens", 0)),
                ),
                request_text=d.get("request_text", ""),
                response_text=d.get("response_text", ""),
                reported_cost_usd=_opt_float(d.get("reported_cost_usd", d.get("spend"))),
            ))
            n += 1
    return n


def _cmd_ingest(args: argparse.Namespace) -> int:
    if not os.path.exists(args.file):
        print(f"file not found: {args.file}", file=sys.stderr)
        return 1
    store = Store(args.db, redact=args.redact)
    if args.format == "litellm":
        n = ingest_litellm_spendlog(args.file, store)
    elif args.format == "bedrock":
        from .connectors import ingest_bedrock_invocation_logs
        n = ingest_bedrock_invocation_logs(args.file, store)
    else:
        n = _ingest_jsonl_native(args.file, store)
    print(f"ingested {n} call(s) into {args.db}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    if not os.path.exists(args.db):
        print(f"db not found: {args.db} (run `retoken ingest` first)", file=sys.stderr)
        return 1
    # Register any rented/flat/per-token models declared in the config so their cost is computed
    # with the right (BOUNDED capacity) model rather than defaulting to per-token PRICING.
    from .costconfig import load_cost_models
    loaded = load_cost_models(args.config)
    if loaded:
        print(f"loaded {loaded} cost model(s) from {args.config}")
    store = Store(args.db)
    print_summary(store)
    if args.html:
        print(f"wrote {write_html(store, args.html)}")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(discrepancy_report_md(store))
        print(f"wrote {args.md}")
    return 0


def _cmd_cockpit(args: argparse.Namespace) -> int:
    from .cockpit import print_cockpit, write_cockpit_html
    from .checkpoint import print_timeline
    print_cockpit(args.manifest)
    print_timeline(args.checkpoints)
    if args.html:
        print(f"wrote {write_cockpit_html(args.manifest, args.html)}")
    return 0


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _cmd_checkpoint(args: argparse.Namespace) -> int:
    from .checkpoint import Checkpoint, add_checkpoint, print_timeline
    if args.add:
        add_checkpoint(Checkpoint(ts=_now(), title=args.add, detail=args.detail or "",
                                  commit=args.commit or "", phase=args.phase or ""),
                       args.checkpoints)
        print(f"checkpoint recorded: {args.add}")
    print_timeline(args.checkpoints)
    return 0


_CAND_REGION_PREFIXES = ("us.", "eu.", "apac.", "us-gov.")


def _parse_candidate(s: str) -> tuple[str, str]:
    """Infer (provider, model) from a candidate id. A dotted vendor-style id (e.g. a Bedrock
    'amazon.nova-pro-v1:0', INCLUDING its ':version' suffix) -> provider from the vendor prefix,
    full id kept as the model. A bare id (e.g. 'gpt-4o-mini') -> provider 'openai'. NB we do NOT
    split on ':' — Bedrock model ids contain a ':version' suffix. Provider only steers tokenizer
    routing; an open-weight model is re-counted via its own tokenizer by model-substring regardless
    of the provider label, so a slightly-off label never mislabels an open-weight count as exact."""
    if "." in s:
        v = s
        for p in _CAND_REGION_PREFIXES:
            if v.startswith(p):
                v = v[len(p):]
                break
        return v.split(".", 1)[0], s
    return "openai", s


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if not os.path.exists(args.db):
        print(f"db not found: {args.db} (run `retoken ingest` first)", file=sys.stderr)
        return 1
    from .costconfig import load_cost_models
    from .evaluator import Candidate, evaluate_migration, render_report, to_dict
    loaded = load_cost_models(args.config)
    if loaded:
        print(f"loaded {loaded} cost model(s) from {args.config}")
    # customer-supplied candidate accept rates: --accept-rate model=0.85
    rates: dict[str, float] = {}
    for pair in (args.accept_rate or []):
        if "=" not in pair:
            print(f"--accept-rate expects model=rate, got {pair!r}", file=sys.stderr)
            return 1
        m, r = pair.rsplit("=", 1)
        try:
            rates[m] = float(r)
        except ValueError:
            print(f"--accept-rate rate must be a number, got {r!r}", file=sys.stderr)
            return 1
    candidates = []
    for c in args.candidate:
        prov, model = _parse_candidate(c)
        candidates.append(Candidate(model=model, provider=prov, accept_rate=rates.get(model)))
    store = Store(args.db)
    workload = store.all_records()
    if not workload:
        print("no calls in the store to evaluate", file=sys.stderr)
        return 1
    report = evaluate_migration(workload, candidates, period_label=args.period or "")
    print(render_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(to_dict(report), f, indent=2)
        print(f"wrote {args.json}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .costconfig import read_config, add_model
    if args.action == "list":
        models = read_config(args.config)
        if not models:
            print(f"no cost models in {args.config} (none registered yet)")
            return 0
        print(f"cost models in {args.config}:")
        for mid, spec in sorted(models.items()):
            extra = " ".join(f"{k}={v}" for k, v in spec.items() if k != "type")
            print(f"  {mid:<28} {spec.get('type','?'):<18} {extra}")
        return 0
    # add — only include values the user actually supplied, so a MISSING required field trips
    # model_from_spec's validation (argparse defaults must not silently fill a required field).
    spec: dict = {"type": args.type}
    if args.type == "per_token":
        for k in ("input", "output", "reasoning", "cache_read"):
            if getattr(args, k) is not None:
                spec[k] = getattr(args, k)
    elif args.type == "flat_subscription":
        if args.fee_per_period is not None:
            spec["fee_per_period"] = args.fee_per_period
    elif args.type == "rented_compute":
        if args.usd_per_gpu_hour is not None:
            spec["usd_per_gpu_hour"] = args.usd_per_gpu_hour
        spec["gpu_count"] = args.gpu_count
        if args.gpu_hours is not None:
            spec["gpu_hours"] = args.gpu_hours
        if args.throughput_tokens_per_hour is not None:
            spec["throughput_tokens_per_hour"] = args.throughput_tokens_per_hour
    try:
        add_model(args.config, args.model_id, spec)
    except ValueError as e:
        print(f"invalid model spec: {e}", file=sys.stderr)
        return 1
    print(f"registered {args.type} model '{args.model_id}' in {args.config}")
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    from .proxy import run_proxy
    run_proxy(args.db, args.upstream, port=args.port, provider=args.provider, host=args.host)
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from .explainer import write_dashboard
    out = write_dashboard(args.out, generated_at=_now(),
                          run_path=args.manifest, checkpoint_path=args.checkpoints)
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="retoken", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="run the offline demo")

    pc = sub.add_parser("cockpit", help="status of every run + resumable commands + progress timeline")
    pc.add_argument("--manifest", default="run-manifest.jsonl")
    pc.add_argument("--checkpoints", default="checkpoints.jsonl")
    pc.add_argument("--html", default=None, help="also write a self-contained HTML cockpit")

    pk = sub.add_parser("checkpoint", help="record a milestone / show the progress timeline")
    pk.add_argument("--add", default=None, help="milestone title to record")
    pk.add_argument("--detail", default=None, help="one-line description")
    pk.add_argument("--commit", default=None, help="short git sha, if committed")
    pk.add_argument("--phase", default=None, help="grouping: research|framework|build|resilience|...")
    pk.add_argument("--checkpoints", default="checkpoints.jsonl")

    pd = sub.add_parser("dashboard", help="regenerate the living architecture+progress dashboard (HTML)")
    pd.add_argument("--out", default="docs/architecture-dashboard.html")
    pd.add_argument("--manifest", default="run-manifest.jsonl")
    pd.add_argument("--checkpoints", default="checkpoints.jsonl")

    pi = sub.add_parser("ingest", help="ingest gateway/provider logs")
    pi.add_argument("file")
    pi.add_argument("--format", choices=["litellm", "bedrock", "jsonl"], default="litellm")
    pi.add_argument("--db", default="retoken.db")
    pi.add_argument("--redact", action="store_true", help="store hashes only, no content at rest")

    pr = sub.add_parser("report", help="print summary + write dashboard/discrepancy report")
    pr.add_argument("--db", default="retoken.db")
    pr.add_argument("--html", default=None)
    pr.add_argument("--md", default=None)
    pr.add_argument("--config", default="retoken-models.json",
                    help="cost-model config (rented/flat models); ignored if absent")

    pe = sub.add_parser("evaluate", help="project the workload onto candidate models (migration eval)")
    pe.add_argument("--db", default="retoken.db")
    pe.add_argument("--candidate", action="append", required=True, metavar="MODEL_ID",
                    help="a candidate model to evaluate (repeatable); e.g. gpt-4o-mini or "
                         "amazon.nova-pro-v1:0 (provider inferred from a dotted vendor prefix)")
    pe.add_argument("--accept-rate", action="append", metavar="model=rate", dest="accept_rate",
                    help="customer-measured accept rate for a candidate (repeatable); e.g. gpt-4o-mini=0.85")
    pe.add_argument("--config", default="retoken-models.json",
                    help="cost-model config (rented/flat models); ignored if absent")
    pe.add_argument("--period", default=None, help="label for the report")
    pe.add_argument("--json", default=None, help="also write the report as JSON")

    pp = sub.add_parser("proxy", help="run the wire-level capture sidecar (passthrough reverse-proxy)")
    pp.add_argument("--upstream", required=True, help="upstream provider base URL, e.g. https://api.openai.com")
    pp.add_argument("--port", type=int, default=8088)
    pp.add_argument("--provider", default="openai", help="provider label for tokenizer routing")
    pp.add_argument("--db", default="retoken.db")
    pp.add_argument("--host", default="127.0.0.1")

    pm = sub.add_parser("models", help="register/list rented/flat/per-token cost models (no code)")
    msub = pm.add_subparsers(dest="action", required=True)
    ml = msub.add_parser("list", help="list registered cost models")
    ml.add_argument("--config", default="retoken-models.json")
    ma = msub.add_parser("add", help="register a cost model")
    ma.add_argument("--config", default="retoken-models.json")
    ma.add_argument("model_id", help="the served model id to attach this cost model to")
    ma.add_argument("--type", required=True,
                    choices=["per_token", "flat_subscription", "rented_compute"])
    # per_token (all four required for this type; default None so a missing one fails validation)
    ma.add_argument("--input", type=float, default=None, help="per_token: USD per 1M input tokens")
    ma.add_argument("--output", type=float, default=None, help="per_token: USD per 1M output tokens")
    ma.add_argument("--reasoning", type=float, default=None, help="per_token: USD per 1M reasoning tokens")
    ma.add_argument("--cache_read", type=float, default=None, help="per_token: USD per 1M cache-read tokens")
    # flat_subscription
    ma.add_argument("--fee_per_period", type=float, default=None, help="flat: fixed fee per period")
    # rented_compute
    ma.add_argument("--usd_per_gpu_hour", type=float, default=None, help="rented: USD per GPU-hour")
    ma.add_argument("--gpu_count", type=int, default=1, help="rented: GPUs provisioned")
    ma.add_argument("--gpu_hours", type=float, default=None,
                    help="rented: provisioned wall-clock hours/period (preferred; exposes idle)")
    ma.add_argument("--throughput_tokens_per_hour", type=float, default=None,
                    help="rented: assumed throughput fallback / for utilisation")

    args = p.parse_args(argv)
    return {"demo": _cmd_demo, "ingest": _cmd_ingest, "report": _cmd_report,
            "cockpit": _cmd_cockpit, "checkpoint": _cmd_checkpoint,
            "dashboard": _cmd_dashboard, "models": _cmd_models,
            "evaluate": _cmd_evaluate, "proxy": _cmd_proxy}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
