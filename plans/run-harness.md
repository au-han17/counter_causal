# plans/run-harness.md

**Branch:** `run-harness` (from `main` @ ec5eb0b)
**runID:** `R-15` — infrastructure, no evaluation. Next free sequence in phase R;
renumber freely if a preliminary block is preferred.

## Hypothesis

Not a hypothesis-driven run. This closes a gap between CLAUDE.md's stated protocol and
what the code can actually do:

- CLAUDE.md requires `configs/<run>.yaml` per run and forbids hand-editing constants in
  source. `evaluate.py` is pure argparse — no config is ever read. R-01/R-02 already
  cite `configs/r01.yaml` / `configs/r02.yaml`, which do not exist.
- CLAUDE.md requires `results/<runID>/manifest.json` recording config path, git commit,
  seed, GPU, wall time and dataset slice. Nothing writes a manifest; nothing captures
  wall time; nothing sets or records a seed.

Without both, runs are not reconstructable and rule 3 cannot be satisfied.

## Protocol

1. New module `runlog.py`; `hooks.py` and `tasks.py` untouched.
2. `--config path.yaml` loads defaults via `parser.set_defaults`; explicit CLI flags win.
3. Unknown YAML keys are a hard error, so typos cannot silently no-op.
4. `--seed` (default 0) seeds python/numpy/torch and is recorded.
5. `--run_id` writes `results/<runID>/{metrics.json,manifest.json,raw/generations.jsonl.gz}`;
   without it, legacy `--out` behaviour is unchanged.
6. Time total wall clock, and hook time separately with `cuda.synchronize` around each call.
7. Record peak GPU memory via `reset_peak_memory_stats` / `max_memory_allocated`.
8. Record sha256 + byte size of the raw dump; warn above 50MB per CLAUDE.md.
9. Add `pyyaml` to `requirements.txt`; raise the `transformers` floor to a version that
   actually has `DynamicCache.layers` and top-level `model.rotary_emb`.
10. Verify with `--single_sample 0` on MATH500; confirm manifest fields and config override.

## Expected runs

No evaluation. Verification only:

| check | command |
|-------|---------|
| config load + override | `--config configs/r01.yaml --single_sample 0` |
| unknown key rejected | config containing a bogus key exits non-zero |
| manifest completeness | every required field non-null after a smoke run |

## Expected outputs

- `runlog.py`
- `configs/r01.yaml`, `configs/r02.yaml` (the two already cited in the ledger)
- `results/R-15/manifest.json` from the smoke run
- Updated `requirements.txt`
- EXPERIMENTS.md row for R-15

## Out of scope

- Any change to eviction, scoring, or generation logic
- Scoring-dtype fix at `hooks.py:451` (owned by `dtype-verification`)
- Per-sample timing inside `tasks.py` — would touch all four task functions; separate plan
