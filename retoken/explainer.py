"""
Living architecture dashboard generator. Writes the plain-language explainer (the six building
blocks, the build assembly line, the resilience story) WITH two data-driven sections injected from
the durable logs: a CHECKPOINT TIMELINE (how things have progressed over time) and a RUN SNAPSHOT
(current jobs). Regenerate any time — `tokenledger dashboard` — and the progress updates.

Self-contained HTML, no JS, no network (NO EGRESS). Placeholders are %%TOKENS%% filled by replace()
so the CSS braces need no escaping.
"""

from __future__ import annotations

import html

from .checkpoint import DEFAULT_PATH as CP_PATH, timeline_html
from .manifest import DEFAULT_PATH as RUN_PATH, load_runs

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TokenLedger — How It All Fits</title>
<style>
  :root{--ink:#15171c;--muted:#6b7280;--line:#e6e8ec;--bg:#f6f7f9;--card:#fff;
    --blue:#2563eb;--blueb:#dbeafe;--green:#15803d;--greenb:#dcfce7;
    --amber:#b45309;--amberb:#fef3c7;--purple:#6d28d9;--purpleb:#ede9fe;}
  *{box-sizing:border-box}
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.5}
  .wrap{max-width:1000px;margin:0 auto;padding:2.2rem 1.2rem 4rem}
  h1{font-size:1.9rem;margin:0 0 .2rem} .sub{color:var(--muted);font-size:1.05rem;margin:0 0 1.6rem}
  h2{font-size:1.15rem;margin:2.2rem 0 .2rem} .lead{color:var(--muted);margin:.1rem 0 1rem}
  .wrap>h3{font-size:1.02rem;margin:1.5rem 0 .4rem;color:#222}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.9rem}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;box-shadow:0 1px 2px rgba(0,0,0,.04)}
  .card h3{margin:.1rem 0 .35rem;font-size:1rem;display:flex;align-items:center;gap:.5rem}
  .card p{margin:.2rem 0;font-size:.92rem} .card .like{color:var(--muted);font-size:.85rem;font-style:italic;margin-top:.5rem}
  .tag{display:inline-block;font-size:.7rem;font-weight:600;padding:.12rem .5rem;border-radius:99px}
  .t-blue{background:var(--blueb);color:var(--blue)} .t-green{background:var(--greenb);color:var(--green)}
  .t-amber{background:var(--amberb);color:var(--amber)} .t-purple{background:var(--purpleb);color:var(--purple)}
  .dot{width:.6rem;height:.6rem;border-radius:50%;display:inline-block}
  .flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:.5rem;margin:.4rem 0}
  .step{flex:1;min-width:120px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:.7rem .8rem}
  .step .n{font-size:.7rem;color:var(--muted);font-weight:700} .step b{display:block;margin:.15rem 0;font-size:.95rem}
  .step span{font-size:.82rem;color:var(--muted)} .arrow{align-self:center;color:#c2c7cf;font-size:1.3rem}
  .callout{background:#0b1020;color:#e7ebf3;border-radius:12px;padding:1.1rem 1.2rem;margin:1.4rem 0}
  .callout code{background:#1c2333;color:#9ecbff;padding:.12rem .45rem;border-radius:5px;font-size:.95rem} .callout h3{margin:.1rem 0 .5rem;color:#fff}
  table{border-collapse:collapse;width:100%;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:.9rem}
  th,td{padding:.5rem .7rem;text-align:left;border-bottom:1px solid var(--line)} th{background:#fafbfc;font-weight:600}
  .s-run{color:var(--blue);font-weight:600} .s-done{color:var(--green)} .s-stop{color:var(--muted)} .s-fail{color:#b91c1c}
  ol.timeline{list-style:none;padding:0;margin:.4rem 0;border-left:2px solid var(--line)}
  ol.timeline li{position:relative;padding:.55rem 0 .55rem 1.2rem;margin-left:.3rem}
  ol.timeline li:before{content:"";position:absolute;left:-7px;top:.95rem;width:11px;height:11px;border-radius:50%;background:var(--green);border:2px solid #fff}
  ol.timeline .when{display:inline-block;font-size:.74rem;color:var(--muted);min-width:8.5rem}
  ol.timeline b{font-size:.95rem} ol.timeline .det{display:block;color:var(--muted);font-size:.85rem;margin-top:.1rem}
  ol.timeline code{background:#f3f3f3;padding:.03rem .35rem;border-radius:4px;font-size:.78rem}
  .foot{color:var(--muted);font-size:.82rem;margin-top:2rem;border-top:1px solid var(--line);padding-top:1rem}
  .pill{font-size:.72rem;color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:.1rem .55rem}
  .stamp{color:var(--muted);font-size:.8rem}
</style></head><body><div class="wrap">

  <h1>TokenLedger — how it all fits</h1>
  <p class="sub">A plain-language map of what we're building, how the AI agents build it, and how it's progressing.</p>
  <p class="stamp">Snapshot generated: %%GENERATED%% &nbsp;·&nbsp; regenerate any time with <code>tokenledger dashboard</code></p>

  <h2>The one-sentence version</h2>
  <p class="lead">Big companies want to use many AI providers (OpenAI, Anthropic, Google…) without being locked to one —
    and without getting overcharged or losing answer quality. <b>TokenLedger is the honest meter and advisor that makes
    that safe:</b> it independently re-checks every AI bill and tracks whether the cheaper option was actually good enough
    to switch to.</p>

  <h2>Progress so far — checkpoints over time</h2>
  <p class="lead">Every milestone is saved to a durable log, so nothing is lost if a limit interrupts us. This timeline is
    built from that log and grows as work proceeds.</p>
  %%TIMELINE%%

  <h2>The building blocks</h2>
  <p class="lead">Six parts. The first three are the product; the last three are how we build and run it.</p>
  <div class="grid">
    <div class="card"><h3><span class="dot" style="background:var(--blue)"></span> The universal plug <span class="tag t-blue">switch layer</span></h3>
      <p>One socket that talks to every AI provider, so you can change provider by flipping a setting — no rewrite, no lock-in. (Santander's <code>llm_bridge</code> pattern.)</p>
      <p class="like">Like a universal travel adapter that works in every country.</p></div>
    <div class="card"><h3><span class="dot" style="background:var(--green)"></span> The honest meter <span class="tag t-green">measure layer</span></h3>
      <p>After every AI call, TokenLedger re-counts the tokens itself and compares to what you were billed — catching over-charges. Nothing leaves your machine.</p>
      <p class="like">Like re-adding the restaurant bill yourself instead of trusting the total.</p></div>
    <div class="card"><h3><span class="dot" style="background:var(--green)"></span> The quality tag <span class="tag t-green">measure layer</span></h3>
      <p>Cost alone is half the story. We also capture “was the answer any good?” (accepted / rejected / score) so we can compare <b>cost per <i>good</i> answer</b>.</p>
      <p class="like">Like noting not just the price of the meal, but whether you’d order it again.</p></div>
    <div class="card"><h3><span class="dot" style="background:var(--purple)"></span> The assembly line <span class="tag t-purple">how we build</span></h3>
      <p>AI agents build on a disciplined line: an architect writes the spec, <b>one</b> builder codes it, independent inspectors review, a final gate approves — then a human merges.</p>
      <p class="like">Like one chef plus separate food inspectors — not five cooks bumping into each other.</p></div>
    <div class="card"><h3><span class="dot" style="background:var(--amber)"></span> The logbook <span class="tag t-amber">resilience</span></h3>
      <p>Every step is written down and saved. If a usage or memory limit cuts us off mid-task, nothing is lost — we reopen the logbook and resume exactly where we stopped.</p>
      <p class="like">Like a ship's logbook: if the lights go out, you still know the exact heading.</p></div>
    <div class="card"><h3><span class="dot" style="background:var(--amber)"></span> The cockpit <span class="tag t-amber">resilience</span></h3>
      <p>A simple board that reads the logbook and shows every job, whether it finished, and a ready-to-paste command to resume anything unfinished.</p>
      <p class="like">Like an airport arrivals board — at a glance, what's done and what's still in the air.</p></div>
  </div>

  <h2>How the agents build it (the assembly line)</h2>
  <p class="lead">This shape isn't guesswork — it's what the experts (Anthropic, Cognition) recommend for letting AI write code reliably. Writing stays with <b>one</b> builder; only the inspectors work in parallel.</p>
  <div class="flow">
    <div class="step"><span class="n">STEP 1 · PLAN</span><b>Spec</b><span>An architect writes exactly what to build and the key decisions — so the builder never improvises.</span></div>
    <div class="arrow">→</div>
    <div class="step"><span class="n">STEP 2 · BUILD</span><b>One builder</b><span>A single agent writes all the code in order and runs the tests. No parallel cooks.</span></div>
    <div class="arrow">→</div>
    <div class="step"><span class="n">STEP 3 · REVIEW</span><b>Inspector panel</b><span>Several independent reviewers (correctness, safety, quality, tests) check it with fresh eyes.</span></div>
    <div class="arrow">→</div>
    <div class="step"><span class="n">STEP 4 · GATE</span><b>Final approval</b><span>A master judge signs off — then a <b>human merges</b> it. Robots never merge unreviewed code.</span></div>
  </div>

  <div class="callout"><h3>👀 Watch it happen live — built into Claude Code</h3>
    <p>You don't have to take my word for what's running. In the Claude Code prompt, type:</p>
    <p style="margin:.6rem 0"><code>/workflows</code> — a live view of every agent, which phase it's in, progress in real time.</p>
    <p>For the durable board that survives a restart (reads the logbook):</p>
    <p style="margin:.6rem 0"><code>tokenledger cockpit --html cockpit.html</code> — all runs + resume commands.</p>
    <p style="margin:.6rem 0"><code>tokenledger dashboard</code> — regenerate THIS page with the latest progress.</p>
  </div>

  <h2>What happens if a limit cuts us off</h2>
  <div class="grid">
    <div class="card"><h3>Usage limit (sudden)</h3><p>A job can stop mid-run. Because each agent's result is saved as it finishes, we re-launch with one command and it <b>replays instantly from where it stopped</b> — already proven this session.</p></div>
    <div class="card"><h3>Memory limit (gradual)</h3><p>The conversation can only hold so much. The real state lives in saved files and commits, so a fresh session re-reads the logbook and continues — no lost progress.</p></div>
  </div>

  <h2 style="border-top:2px solid var(--line);padding-top:1.6rem;margin-top:2.6rem">Go to market — in plain language</h2>
  <p class="lead">Who buys it, how we sell it, and what we build next — explained simply.</p>

  <h3>1 &middot; Who actually buys it</h3>
  <p class="lead">There is no single boss to sell to. An engineer who feels the pain <b>champions</b> it,
  then carries a one-page case <b>up to a committee</b> (finance + FinOps + security) who decide together.</p>
  <div class="flow">
    <div class="step"><span class="n">THE CHAMPION</span><b>Engineer / platform lead</b><span>Feels the bill pain, runs the tool. Can recommend — can't sign.</span></div>
    <div class="arrow">&rarr;</div>
    <div class="step"><span class="n">THE PACKAGE</span><b>One-page business case</b><span>Savings + risk + price. The champion forwards it up.</span></div>
    <div class="arrow">&rarr;</div>
    <div class="step"><span class="n">THE COMMITTEE</span><b>CFO &middot; FinOps &middot; Security</b><span>Decide together; one of them signs.</span></div>
  </div>
  <p class="lead" style="font-size:.85rem">Research-verified: FinOps is NOT a single gatekeeper — it advises; engineers pick the model; a committee decides.</p>

  <h3>2 &middot; We tell two stories</h3>
  <div class="grid">
    <div class="card"><h3><span class="tag t-purple">to the champion (engineer)</span></h3>
      <p>"Install it in an afternoon. It reads your existing logs — change nothing, nothing leaves your network.
      In a week you'll see where money is wasted and which jobs can move to cheaper models at <i>your</i> quality
      bar. Walk into the cost review with proof."</p></div>
    <div class="card"><h3><span class="tag t-blue">to the committee (the bosses)</span></h3>
      <p>"Your AI spend is a black box — one supplier, numbers you can't check. We independently verify what you
      spend and move the right jobs to cheaper models <i>inside your existing cloud</i> — 40-60% cheaper, without
      losing quality. Self-hosted, nothing leaves the building."</p></div>
  </div>

  <h3>3 &middot; Who hurts the most — who we help first</h3>
  <div class="grid">
    <div class="card"><h3>&#128296; Coding &amp; agent tools <span class="tag t-green">help first</span></h3>
      <p>They resell AI to developers. When the AI provider raised prices and added limits, their costs jumped and
      margins were crushed.</p><p class="like">Proven pain — and our tool works on their setup today.</p></div>
    <div class="card"><h3>&#128201; Thin-margin AI apps <span class="tag t-green">help first</span></h3>
      <p>Fixed-price contracts, but the AI bill scales with usage. One price hike eats the profit.</p>
      <p class="like">Inference is ~a quarter of their revenue; we cut it and lock it down.</p></div>
    <div class="card"><h3>&#127974; Regulated / big enterprise <span class="tag t-amber">scale later</span></h3>
      <p>On a hyperscaler (AWS/Azure/Google); data can't leave, locked in by big spend commitments.</p>
      <p class="like">"Switch the model inside your cloud" — change a model ID, keep everything else.</p></div>
  </div>

  <div class="callout"><h3>4 &middot; How we prove it — the 7-day test</h3>
    <p style="margin:.5rem 0"><b>Day 0</b> — plug our sidecar into their system (their keys, their network, nothing leaves).</p>
    <p style="margin:.5rem 0"><b>Days 1-7</b> — it quietly watches real traffic. They do nothing.</p>
    <p style="margin:.5rem 0"><b>Day 7</b> — we show three things: (1) "your bill, independently re-counted", (2) "cost per <i>good</i>
    answer, by task", (3) "move these jobs, save $X — here's the break-even".</p>
    <p style="margin:.5rem 0">&#9989; <b>Success = they ask "how much does it cost?" without being prompted.</b></p>
  </div>

  <h3>5 &middot; What we build next — in order</h3>
  <p class="lead">Corrected after review: prove the value first with the segment in most pain, then scale.</p>
  <div class="flow">
    <div class="step"><span class="n">STEP 1</span><b>Flexible pricing model</b><span>Handle pay-per-token AND rented/flat models (groundwork).</span></div>
    <div class="arrow">&rarr;</div>
    <div class="step"><span class="n">STEP 2 &middot; prove it</span><b>Migration adviser</b><span>"Move this job to that model, save $X." Test with a coding-tool partner — works today.</span></div>
    <div class="arrow">&rarr;</div>
    <div class="step"><span class="n">STEP 3 &middot; scale</span><b>AWS Bedrock connector</b><span>Plug into big enterprises; map each job to a cheaper in-cloud model.</span></div>
  </div>

  <h3>6 &middot; Why we win</h3>
  <div class="grid">
    <div class="card"><h3>&#128274; Your data never leaves</h3><p>We check your bill without sending your prompts anywhere. Most SaaS tools upload your data to their cloud — we don't.</p></div>
    <div class="card"><h3>&#9989; We verify, not trust</h3><p>Everyone else repeats the provider's numbers. We re-count them ourselves.</p></div>
    <div class="card"><h3>&#128260; Switch without the pain</h3><p>Move a job to a cheaper model inside the cloud you already use — like changing a setting, not rebuilding.</p></div>
  </div>

  <h2>Current jobs</h2>
  <p class="lead">From the run log. For the live, always-current version use <code>/workflows</code>.</p>
  <table><tr><th>Job</th><th>Status</th><th>Phase</th></tr>%%RUNS%%</table>

  <p class="foot">
    <span class="pill">no data leaves your machine</span>
    <span class="pill">honest labels: exact / estimated / unverifiable</span>
    <span class="pill">a human approves every merge</span><br>
    Full technical detail in <code>README.md</code>.
  </p>
</div></body></html>"""


def _runs_rows(run_path: str) -> str:
    cls = {"running": "s-run", "completed": "s-done", "stopped": "s-stop", "failed": "s-fail"}
    mark = {"running": "● running", "completed": "✓ done", "stopped": "■ stopped", "failed": "✕ failed"}
    rows = []
    for r in load_runs(run_path):
        c = cls.get(r.status, "")
        rows.append(
            f"<tr><td>{html.escape(r.name)}</td>"
            f"<td class='{c}'>{html.escape(mark.get(r.status, r.status))}</td>"
            f"<td>{html.escape(r.phase)}</td></tr>"
        )
    return "".join(rows) or "<tr><td colspan='3'>no runs recorded</td></tr>"


def write_dashboard(out: str = "docs/architecture-dashboard.html", *, generated_at: str = "",
                    run_path: str = RUN_PATH, checkpoint_path: str = CP_PATH) -> str:
    doc = (_TEMPLATE
           .replace("%%GENERATED%%", html.escape(generated_at or "(unspecified)"))
           .replace("%%TIMELINE%%", timeline_html(checkpoint_path))
           .replace("%%RUNS%%", _runs_rows(run_path)))
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return out
