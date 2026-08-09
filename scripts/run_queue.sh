#!/usr/bin/env bash
# Run queue for the reduced reproduction set (6 runs).
#
#   bash scripts/run_queue.sh --preflight        # checks only
#   bash scripts/run_queue.sh --parallel 4 math  # the 4 MATH500 runs at once
#   bash scripts/run_queue.sh --parallel 2 lh    # the 2 LongHealth runs at once
#   bash scripts/run_queue.sh R-06               # one run, output on the terminal
#   bash scripts/run_queue.sh --finalize         # ledger + commit + push what finished
#
# Groups: `math` = R-00-qwen R-06 R-09 R-10, `lh` = R-11-full R-11-fast, `all` = both.
#
# Parallel mode gives every run its own logs/<runID>.log and prints only one line per
# state change, so four concurrent runs stay readable. Follow one with:
#     tail -f logs/R-06.log
#
# Git is touched only by --finalize, which you run yourself after reviewing the results —
# concurrent commits would collide on index.lock. Resumable: a run whose
# results/<runID>/metrics.json exists is skipped.
#
# wall_sec / peak_gpu_gb are still recorded but are NOT meaningful under --parallel,
# since runs contend for the same GPU. Accuracy is unaffected.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-$(command -v python || command -v python3)}"
mkdir -p logs
LOG="logs/queue-$(date +%Y%m%d-%H%M%S)-$$.log"

MATH_RUNS=(
  "R-00-qwen:configs/r00_qwen.yaml"
  "R-06:configs/r06.yaml"
  "R-09:configs/r09.yaml"
  "R-10:configs/r10.yaml"
)
LH_RUNS=(
  "R-11-full:configs/r11_full.yaml"
  "R-11-fast:configs/r11_fast.yaml"
  "R-11-none:configs/r11_none.yaml"
  "R-11-h2o:configs/r11_h2o.yaml"
)
# deferred reproduction, resumed while report drafting (use --parallel 4)
MATH2_RUNS=(
  "R-00-llama:configs/r00_llama.yaml"
  "R-01:configs/r01.yaml"
  "R-02:configs/r02.yaml"
  "R-03:configs/r03.yaml"
  "R-04:configs/r04.yaml"
  "R-05:configs/r05.yaml"
  "R-07:configs/r07.yaml"
  "R-08:configs/r08.yaml"
)
# LoCoMo multi-hop at cache 15000 (use --parallel 2 — counter's scoring pass over
# ~19k candidates peaks ~35GB; the order below keeps heavy runs from pairing)
R13_RUNS=(
  "R-13-full:configs/r13_full.yaml"
  "R-13-sliding:configs/r13_sliding.yaml"
  "R-13-h2o:configs/r13_h2o.yaml"
  "R-13-counter:configs/r13_counter.yaml"
  "R-13-fast:configs/r13_fast.yaml"
)
RUNS=("${MATH_RUNS[@]}" "${LH_RUNS[@]}" "${MATH2_RUNS[@]}" "${R13_RUNS[@]}")

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

preflight() {
  local ok=0
  say "=== preflight ==="

  if ! git config user.email >/dev/null || ! git config user.name >/dev/null; then
    say "FAIL git identity unset. Fix:"
    say "     git config user.name 'au-han17'; git config user.email 'khan17wong@gmail.com'"
    ok=1
  else
    say "ok   git identity: $(git config user.name) <$(git config user.email)>"
  fi

  if [ "$PUSH" = "0" ]; then
    say "skip push auth (--no-push)"
  elif git push --dry-run origin HEAD >>"$LOG" 2>&1; then
    say "ok   push auth"
  else
    say "FAIL cannot push. Bake a token into the remote, or pass --no-push:"
    say "     git remote set-url origin https://<TOKEN>@github.com/au-han17/counter_causal.git"
    ok=1
  fi

  [ -n "$(git status --porcelain)" ] && say "WARN tree dirty — manifests record dirty:true" \
                                     || say "ok   working tree clean"

  say "     checking torch/cuda/yaml — first import can take 30-90s ..."
  if $PY -c "import torch,yaml,transformers; assert torch.cuda.is_available()" 2>>"$LOG"; then
    say "ok   torch+cuda+yaml: $($PY -c 'import torch;print(torch.cuda.get_device_name(0))')"
  else
    say "FAIL torch/cuda/yaml import, or no GPU visible"
    ok=1
  fi

  if $PY tests/validate_configs.py >>"$LOG" 2>&1; then
    say "ok   configs validate"
  else
    say "FAIL config validation — see $LOG"
    ok=1
  fi

  say "     HF_HOME=${HF_HOME:-<unset, will use container disk!>}"
  return $ok
}

run_one() {
  local id="$1" cfg="$2" runlog="logs/$1.log"

  if [ -f "results/$id/metrics.json" ]; then
    say "skip $id (already complete)"
    return 0
  fi

  local t0=$SECONDS failed=0
  say "START $id  ($cfg)  -> $runlog"

  if [ "$QUIET" = "1" ]; then
    $PY evaluate.py --config "$cfg" --run_id "$id" >"$runlog" 2>&1 || failed=1
  else
    $PY evaluate.py --config "$cfg" --run_id "$id" 2>&1 | tee "$runlog"
    failed=${PIPESTATUS[0]}
  fi

  if [ "$failed" = "0" ]; then
    local score
    score=$($PY -c "import json;d=json.load(open('results/$id/metrics.json'));print(f\"{d['score_key']}={d['score']:.4f}\")" 2>/dev/null || echo "score=?")
    say "DONE  $id  $score  ($((SECONDS - t0))s)"
  else
    say "FAIL  $id after $((SECONDS - t0))s — see $runlog"
  fi

  if [ "$COMMIT" = "1" ]; then
    commit_one "$id" "$failed"
  fi
  return 0
}

commit_one() {
  local id="$1" failed="${2:-0}"
  if [ "$failed" != "0" ]; then
    $PY scripts/ledger_update.py "$id" --failed "run exited non-zero; see logs/$id.log" >>"$LOG" 2>&1
  else
    $PY scripts/ledger_update.py "$id" >>"$LOG" 2>&1
  fi
  git add results EXPERIMENTS.md 2>/dev/null
  git diff --cached --quiet && return 0
  git commit -q -m "[$id] run result" && say "committed $id"
  if [ "$PUSH" = "1" ]; then
    git push -q origin HEAD 2>>"$LOG" && say "pushed $id" || say "WARN push failed for $id"
  fi
}

finalize() {
  say "=== finalize: ledger + commit + push for every completed run ==="
  local n=0
  for entry in "${RUNS[@]}"; do
    local id="${entry%%:*}"
    [ -f "results/$id/metrics.json" ] || continue
    commit_one "$id" 0 && n=$((n + 1))
  done
  say "finalize done ($n run(s) considered)"
  grep -E "^\| R-" EXPERIMENTS.md | tee -a "$LOG"
}

# ---- entry ----
PUSH=1; COMMIT=1; QUIET=0; PARALLEL=1
PREFLIGHT_ONLY=0; FINALIZE_ONLY=0
WANTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --no-push)   PUSH=0 ;;
    --no-commit) COMMIT=0 ;;
    --quiet)     QUIET=1 ;;
    --preflight) PREFLIGHT_ONLY=1 ;;
    --finalize)  FINALIZE_ONLY=1 ;;
    --parallel)  shift; PARALLEL="${1:-1}" ;;
    math)        for e in "${MATH_RUNS[@]}"; do WANTED+=("${e%%:*}"); done ;;
    lh)          for e in "${LH_RUNS[@]}";   do WANTED+=("${e%%:*}"); done ;;
    math2)       for e in "${MATH2_RUNS[@]}"; do WANTED+=("${e%%:*}"); done ;;
    r13)         for e in "${R13_RUNS[@]}";  do WANTED+=("${e%%:*}"); done ;;
    all)         for e in "${RUNS[@]}";      do WANTED+=("${e%%:*}"); done ;;
    -*)          say "unknown option: $1"; exit 2 ;;
    *)           WANTED+=("$1") ;;
  esac
  shift
done

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  if preflight; then say "preflight OK"; exit 0; else say "preflight FAILED"; exit 1; fi
fi
if [ "$FINALIZE_ONLY" = "1" ]; then finalize; exit 0; fi

# Parallel streams must not touch git; --finalize does it afterwards, sequentially.
if [ "$PARALLEL" -gt 1 ]; then
  COMMIT=0
  QUIET=1
  say "parallel=$PARALLEL — per-run logs, git deferred to the finalize pass"
fi

preflight || { say "preflight FAILED — aborting before any GPU time is spent"; exit 1; }

QUEUE=()
if [ ${#WANTED[@]} -gt 0 ]; then
  for want in "${WANTED[@]}"; do
    for entry in "${RUNS[@]}"; do
      [ "${entry%%:*}" = "$want" ] && QUEUE+=("$entry")
    done
  done
else
  QUEUE=("${RUNS[@]}")
fi

say "queue: ${#QUEUE[@]} run(s), parallel=$PARALLEL"

if [ "$PARALLEL" -gt 1 ]; then
  running=0
  for entry in "${QUEUE[@]}"; do
    run_one "${entry%%:*}" "${entry#*:}" &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
  done
  wait
  say "all streams finished. Nothing was committed — review, then run:"
  say "    bash scripts/run_queue.sh --finalize"
else
  for entry in "${QUEUE[@]}"; do
    run_one "${entry%%:*}" "${entry#*:}"
  done
fi

say "=== queue finished; log: $LOG ==="
