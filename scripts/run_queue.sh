#!/usr/bin/env bash
# Sequential run queue: execute -> update ledger -> commit -> push, one run at a time.
#
#   bash scripts/run_queue.sh              # everything still outstanding
#   bash scripts/run_queue.sh R-00-llama   # just these
#   bash scripts/run_queue.sh --preflight  # checks only, run nothing
#   bash scripts/run_queue.sh --no-push    # commit locally, never push
#
# --no-push still commits every run, so nothing is lost to a crash — only to losing the
# pod itself. Push the lot afterwards with:  git push origin main
#
# Resumable: any run with results/<runID>/metrics.json already present is skipped, so a
# reclaimed pod costs at most the in-flight run. A failing run is logged and the queue
# continues. Runs are strictly sequential — the manifests record wall time and peak GPU
# memory, which concurrent runs would make meaningless.

set -uo pipefail   # deliberately not -e: one bad run must not kill the queue

cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-$(command -v python || command -v python3)}"
mkdir -p logs
LOG="logs/queue-$(date +%Y%m%d-%H%M%S).log"

# runID:config — R-00-llama first as the behaviour check
RUNS=(
  "R-00-llama:configs/r00_llama.yaml"
  "R-00-qwen:configs/r00_qwen.yaml"
  "R-01:configs/r01.yaml"
  "R-02:configs/r02.yaml"
  "R-03:configs/r03.yaml"
  "R-04:configs/r04.yaml"
  "R-05:configs/r05.yaml"
  "R-06:configs/r06.yaml"
  "R-07:configs/r07.yaml"
  "R-08:configs/r08.yaml"
  "R-09:configs/r09.yaml"
  "R-10:configs/r10.yaml"
  "R-11-full:configs/r11_full.yaml"
  "R-11-fast:configs/r11_fast.yaml"
  "R-12-full:configs/r12_full.yaml"
  "R-12-fast:configs/r12_fast.yaml"
  "R-13-full:configs/r13_full.yaml"
  "R-13-sliding:configs/r13_sliding.yaml"
  "R-13-h2o:configs/r13_h2o.yaml"
  "R-13-counter:configs/r13_counter.yaml"
  "R-13-fast:configs/r13_fast.yaml"
  "R-14-full:configs/r14_full.yaml"
  "R-14-sliding:configs/r14_sliding.yaml"
  "R-14-h2o:configs/r14_h2o.yaml"
  "R-14-counter:configs/r14_counter.yaml"
  "R-14-fast:configs/r14_fast.yaml"
)

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

preflight() {
  local ok=0
  say "=== preflight ==="

  if ! git config user.email >/dev/null || ! git config user.name >/dev/null; then
    say "FAIL git identity unset. Fix:"
    say "     git config user.name 'uakhan17'; git config user.email 'khan17wong@gmail.com'"
    ok=1
  else
    say "ok   git identity: $(git config user.name) <$(git config user.email)>"
  fi

  if [ "$PUSH" = "0" ]; then
    say "skip push auth (--no-push); results commit locally only"
  elif git push --dry-run origin HEAD >>"$LOG" 2>&1; then
    say "ok   push auth"
  else
    say "FAIL cannot push. Either bake a token into the remote:"
    say "     git remote set-url origin https://<TOKEN>@github.com/au-han17/counter_causal.git"
    say "     or re-run with --no-push to commit locally and push in the morning."
    ok=1
  fi

  if [ -n "$(git status --porcelain)" ]; then
    say "WARN working tree dirty — manifests will record dirty:true"
  else
    say "ok   working tree clean"
  fi

  say "     checking torch/cuda/yaml — first import on a fresh pod can take 30-90s ..."
  if $PY -c "import torch,yaml,transformers; assert torch.cuda.is_available()" 2>>"$LOG"; then
    say "ok   torch+cuda+yaml: $($PY -c 'import torch;print(torch.cuda.get_device_name(0))')"
  else
    say "FAIL torch/cuda/yaml import or no GPU visible"
    ok=1
  fi

  if $PY tests/validate_configs.py >>"$LOG" 2>&1; then
    say "ok   configs validate"
  else
    say "FAIL config validation — see $LOG"
    ok=1
  fi

  return $ok
}

run_one() {
  local id="$1" cfg="$2"

  if [ -f "results/$id/metrics.json" ]; then
    say "skip $id (already complete)"
    return 0
  fi

  say "---- $id  ($cfg) ----"
  local t0=$SECONDS
  if $PY evaluate.py --config "$cfg" --run_id "$id" 2>&1 | tee -a "$LOG"; then
    say "done $id in $((SECONDS - t0))s"
    $PY scripts/ledger_update.py "$id" 2>&1 | tee -a "$LOG"
  else
    say "FAIL $id after $((SECONDS - t0))s — continuing"
    $PY scripts/ledger_update.py "$id" --failed "run exited non-zero; see $LOG" 2>&1 | tee -a "$LOG"
  fi

  git add results EXPERIMENTS.md 2>/dev/null
  if git diff --cached --quiet; then
    say "warn $id produced nothing to commit"
    return 0
  fi
  git commit -q -m "[$id] run result" && say "committed $id"
  if [ "$PUSH" = "0" ]; then
    say "not pushed ($id) — --no-push"
  elif git push -q origin HEAD 2>>"$LOG"; then
    say "pushed $id"
  else
    say "WARN push failed for $id — results are committed locally only"
  fi
}

# ---- entry ----
PUSH=1
PREFLIGHT_ONLY=0
WANTED=()
for a in "$@"; do
  case "$a" in
    --no-push)   PUSH=0 ;;
    --preflight) PREFLIGHT_ONLY=1 ;;
    -*)          say "unknown option: $a"; exit 2 ;;
    *)           WANTED+=("$a") ;;
  esac
done

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  if preflight; then say "preflight OK"; exit 0; else say "preflight FAILED"; exit 1; fi
fi

preflight || { say "preflight FAILED — aborting before any GPU time is spent"; exit 1; }
[ "$PUSH" = "0" ] && say "NOTE --no-push: run 'git push origin main' once you are back"

if [ ${#WANTED[@]} -gt 0 ]; then
  for want in "${WANTED[@]}"; do
    for entry in "${RUNS[@]}"; do
      [ "${entry%%:*}" = "$want" ] && run_one "${entry%%:*}" "${entry#*:}"
    done
  done
else
  for entry in "${RUNS[@]}"; do
    run_one "${entry%%:*}" "${entry#*:}"
  done
fi

say "=== queue finished; log: $LOG ==="
