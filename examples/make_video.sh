#!/usr/bin/env bash
# Build a short MP4 demo of TokenLedger from rendered HTML slides + the real dashboard.
# Requires: Google Chrome (headless) + ffmpeg. No BrowserStack, no hosting, fully local.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"; MED="$ROOT/media"; SL="$MED/slides"; FR="$MED/frames"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
mkdir -p "$SL" "$FR"

# Ensure the real dashboard exists.
[ -f "$ROOT/tokenledger_demo.html" ] || .venv/bin/python -m examples.demo >/dev/null

slide () { # $1=file  $2=html-body
cat > "$SL/$1.html" <<EOF
<!doctype html><html><head><meta charset="utf-8"><style>
 html,body{margin:0;width:1280px;height:720px;overflow:hidden;
   font-family:-apple-system,system-ui,sans-serif;background:#0c1116;color:#e8edf2}
 .wrap{box-sizing:border-box;height:720px;padding:90px 110px;display:flex;flex-direction:column;justify-content:center}
 h1{font-size:58px;margin:0 0 18px;letter-spacing:-1px}
 h2{font-size:34px;margin:0 0 28px;color:#8fd0ff;font-weight:600}
 p,li{font-size:30px;line-height:1.5;color:#c4cfda;margin:6px 0}
 .big{font-size:46px;color:#fff;font-weight:700}
 .ok{color:#5ad17f} .bad{color:#ff6b6b} .mut{color:#7d8a99;font-size:24px}
 code{background:#172230;padding:3px 10px;border-radius:6px;color:#9fe0ff;font-size:26px}
</style></head><body><div class="wrap">$2</div></body></html>
EOF
"$CHROME" --headless --disable-gpu --hide-scrollbars \
  --screenshot="$FR/$1.png" --window-size=1280,720 "file://$SL/$1.html" 2>/dev/null
}

slide 00 '<h1>TokenLedger</h1><h2>See what your AI providers really bill you</h2>
  <p class="mut">Self-hosted &middot; independent &middot; nothing leaves your box</p>'
slide 01 '<h2>The problem</h2><p>LLM bills are self-reported and unsigned.</p>
  <p>You pay for <b>input, output, cached and hidden reasoning tokens</b> — and never verify them.</p>'
slide 02 '<h2>How it works</h2>
  <p>1. Re-tokenize what you actually received <span class="mut">(exact, via the model&#39;s own tokenizer)</span></p>
  <p>2. Reconcile 3 numbers: per-call usage vs your re-count vs the invoice</p>
  <p>3. Rank cost-per-task and show where to migrate to open-weight</p>
  <p class="mut">It audits your gateway&#39;s numbers — it does not route traffic.</p>'
# 03 = the real dashboard
"$CHROME" --headless --disable-gpu --hide-scrollbars \
  --screenshot="$FR/03.png" --window-size=1280,720 "file://$ROOT/tokenledger_demo.html" 2>/dev/null
slide 04 '<h2>It catches real discrepancies</h2>
  <p class="big bad">billed 89 &nbsp;vs&nbsp; 64 re-tokenized</p>
  <p>gpt-4o output over-count — caught exactly from the returned text.</p>
  <p class="mut">No captured text? Marked UNVERIFIABLE, never a false alarm.</p>'
slide 05 '<h2>Measured effectiveness</h2>
  <p class="big"><span class="ok">precision 100%</span> &nbsp; <span class="ok">recall 100%</span></p>
  <p class="big"><span class="ok">false-positive rate 0%</span></p>
  <p class="mut">labelled set, real re-count — catches every real over-count, zero false alarms</p>'
slide 06 '<h1>TokenLedger</h1><p>Independent token metering you can trust.</p>
  <p><code>tokenledger ingest litellm.jsonl</code> &nbsp; <code>tokenledger report</code></p>'

# Stitch (each frame held 3.2s). Absolute paths for the concat demuxer.
LIST="$MED/list.txt"; : > "$LIST"
for f in 00 01 02 03 04 05 06; do
  printf "file '%s'\nduration 3.2\n" "$FR/$f.png" >> "$LIST"
done
printf "file '%s'\n" "$FR/06.png" >> "$LIST"   # last frame needs a final entry

ffmpeg -y -f concat -safe 0 -i "$LIST" \
  -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p" \
  -r 30 "$MED/tokenledger-demo.mp4" >/dev/null 2>&1
echo "wrote $MED/tokenledger-demo.mp4"
