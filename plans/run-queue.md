# plans/run-queue.md

**Branch:** `run-queue` (from `main` @ 61eef61)
**runID:** `R-16` — infrastructure, no evaluation

## Hypothesis

Not hypothesis-driven. CLAUDE.md rule 3 requires that after *every* run the manifest is
written, the ledger row appended, and the result committed and pushed — and the pod is
ephemeral, so "an unpushed result does not exist". Today only the manifest is automatic;
the ledger row, commit and push are manual. An unattended overnight sweep would therefore
produce 26 runs of results that vanish with the pod.

## Protocol

1. `scripts/ledger_update.py <runID>` reads `results/<runID>/{metrics,manifest}.json` and
   fills date / commit / GPU-h / status / headline into the matching EXPERIMENTS.md row,
   appending a new row when no exact runID row exists (R-06..R-10, R-11-full, …).
2. Failed runs get a row with status `FAILED` plus a Failure-log entry, per rule 3.
3. `scripts/run_queue.sh` runs the 26 configs sequentially on one GPU.
4. Resumable: any run whose `results/<runID>/metrics.json` exists is skipped, so a
   reclaimed pod loses at most the in-flight run.
5. Preflight before the queue starts: git identity, push auth, CUDA visible, imports OK.
   Fail loudly up front rather than after hours of compute.
6. Per run: execute -> update ledger -> `git add results/ EXPERIMENTS.md` -> commit
   `[runID] ...` -> push. One run = one row = one commit.
7. A single failing run must not abort the queue; log and continue.
8. All output tee'd to `logs/queue-<n>.log`.

## Expected runs

No evaluation of its own. Drives R-00-llama, R-00-qwen, R-01..R-10, R-11-{full,fast},
R-12-{full,fast}, R-13-*, R-14-* — 26 runs, R-00-llama first as the behaviour check.

## Expected outputs

- `scripts/run_queue.sh`, `scripts/ledger_update.py`
- One commit + push per completed run
- EXPERIMENTS.md rows filled in automatically

## Out of scope

- Any change to eviction, scoring or generation logic
- Parallel execution: manifests record `wall_sec` and `peak_gpu_gb`, which concurrent
  runs would render meaningless
