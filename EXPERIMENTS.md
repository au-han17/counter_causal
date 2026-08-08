# EXPERIMENTS.md — Run Ledger

Append-only. One row per launched run, including failures and aborted runs.
Run ID scheme: `<phase>-<seq>` where phase ∈ {R (reproduction), Q1, Q2, Q3}.
Every row must have a matching `results/<runID>/manifest.json` and a commit tagged
`[runID]` on the branch named in the row.

## Status board (update as you go)
- [ ] Pod env verified (R-01, R-02 within tolerance)
- [ ] MATH500 reproduction complete (R-01 … R-10)
- [ ] Prefill-heavy anomaly points reproduced (R-11 … R-14)
- [ ] Q1 sweep done, best layer identified
- [ ] Q1 validation run at LongHealth anomaly config
- [ ] Q2 unseen attention pattern in flipped mask forward pass
- [ ] Q3 surprise score quality drift, staleness 
- [ ] Report drafted

## Preliminary Precision Verification
In the original repo, model weights loaded for Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct are in float16. However, their original weights are in bf16 format. Rerun two models in bf16 on Math500 and compare results with reference numbers in the paper. If results deviates, make new plans. potential surprise score dtype issue: counter causal in fp32; counter causal fast in fp16 (pending on model weights dtype).

**Tracked as R-00** — branch `dtype-verification`, plan `plans/dtype-verification.md`.
Both checkpoints confirmed `torch_dtype: bfloat16` in their HF `config.json`
(Llama's own repo is gated; value read from the `unsloth` mirror of the same weights).
`evaluate.py` now takes `--dtype {float16,bfloat16,float32}`, defaulting to `float16`
so nothing already run changes silently. Weight dtype is recorded in each run's output JSON.

Note the direction of the scoring-dtype interaction: `counter_causal_fast_hook` never
casts its logits (`hooks.py:451`) while `counter_causal_hook` casts to fp32
(`hooks.py:357`). bf16 has 7 mantissa bits against fp16's 10, so moving weights to bf16
makes the fast hook's surprise *ranking* coarser, not finer. R-05/R-10 therefore confound
two effects and should not be read as a clean weight-dtype signal; R-01/R-06 (no eviction)
are the clean comparisons. Scoring dtype left unchanged pending those results.

## Run Harness (R-15)
Branch `run-harness`, plan `plans/run-harness.md`. Closes the gap between the protocol
above and what the code could do: `evaluate.py` read no config, wrote no manifest, and
captured no seed or wall time, so rule 3 was unsatisfiable. Adds `--config/--run_id/--seed`
plus `runlog.py` (config loader, seeding, GPU-synchronised hook timing, manifest writer).

**Environment floor corrected.** `requirements.txt` pinned `transformers>=4.40.0`, which
cannot run this code: `hooks.py` uses `DynamicCache.layers` / `DynamicLayer` (added 4.54)
and `evaluate.py` calls `from_pretrained(dtype=...)` (replaced `torch_dtype` in 4.56).
Verified by inspecting `cache_utils.py` and `modeling_utils.py` at tags v4.40.0/v4.54.0/
v4.56.0. Floor raised to `>=4.56.0`; `pyyaml` added. Pin the exact resolved versions on
the pod and record them in each manifest — `env.transformers` is captured automatically.

## Branch layout
`main` = upstream method + our instrumentation. `run-harness` and `dtype-verification`
are merged in; the integration branch `r00-bf16` is merged and deleted. Defaults are
unchanged (`--dtype float16`, `--run_id` opt-in), so `main` still reproduces Steve's
behaviour exactly — it can now record what it did. Nothing in `hooks.py` or `tasks.py`
was touched, so rule 1 holds.

All reproduction runs (R-00 … R-15) execute from `main` with `--config` + `--run_id`.
Q1/Q2/Q3 branch from `main` and inherit the harness. The original `run-harness` and
`dtype-verification` branches are kept as the record of each change in isolation.

Every `configs/*.yaml` is checked against the live argparse definition by
`tests/validate_configs.py` (26/26 valid). `recent_size` is deliberately omitted
everywhere so `build_hook` applies the upstream defaults documented in README
"Key arguments": 0, or `cache_size//2` for h2o.

## Ledger

| runID | date | branch | commit | model | dataset (slice) | strategy | config | GPU-h | status | headline result | notes |
|-------|------|--------|--------|-------|-----------------|----------|--------|-------|--------|-----------------|-------|
| R-00 | 2026-08-08 | dtype-verification | ca3919a | both | n/a | n/a (code change) | plans/dtype-verification.md | 0 | done | upstream loads both checkpoints as fp16; both are natively bf16 | no evaluation run; adds `--dtype`, default float16 so R-runs are unchanged unless passed. Verification carried by R-01/R-05/R-06/R-10 re-run with `--dtype bfloat16` |
| R-00-qwen | | main | | Qwen2.5-7B | MATH500 (500) | full | configs/r00_qwen.yaml | | queued | | bf16 vs upstream fp16; target ≈.766; `hook: none` isolates weight dtype |
| R-00-llama | | main | | Llama-3.1-8B | MATH500 (500) | full | configs/r00_llama.yaml | | queued | | bf16 vs upstream fp16; target ≈.488; gated model, needs HF login |
| R-01 | | main | | Llama-3.1-8B | MATH500 | full | configs/r01.yaml | | | | env sanity anchor; target ≈.488 |
| R-02 | | main | | Llama-3.1-8B | MATH500 | sliding | configs/r02.yaml | | | | target ≈.458 |
| R-03 | | main | | Llama-3.1-8B | MATH500 | H2O | configs/r03.yaml | | | | target ≈.464 |
| R-04 | | main | | Llama-3.1-8B | MATH500 | counter-causal (full) | configs/r04.yaml | | | | target ≈.482; enable --dump_scores |
| R-05 | | main | | Llama-3.1-8B | MATH500 | counter-causal (fast) | configs/r05.yaml | | | | target ≈.480 |
| R-06..R-10 | | main | | Qwen2.5-7B | MATH500 | same five | configs/r06..r10.yaml | | | | targets .766/.692/.762/.744/.736 |
| R-11 | | main | | Llama-3.1-8B | LongHealth (all 400) | counter-causal (full) vs (fast) | configs/r11_{full,fast}.yaml (cache 8000) | | | | anomaly: counter-causal fast > counter-causal full? |
| R-12 | | main | | Llama-3.1-8B | LongHealth | same | configs/r12_{full,fast}.yaml (cache 9000) | | | | robustness of anomaly |
| R-13 | | main | | Llama-3.1-8B | LoCoMo multi-hop (282) | five strategies | configs/r13_*.yaml (cache 15000) | | | | anomaly: fast > full-cache? H2O collapse? |
| R-14 | | main | | Llama-3.1-8B | LoCoMo multi-hop (282) | five strategies | configs/r14_*.yaml (cache 17000) | | | | robustness |
| R-15 | 2026-08-08 | run-harness | 5507952 | n/a | n/a | n/a (infrastructure) | plans/run-harness.md | 0 | code done, smoke run pending | config loader + manifest writer; `transformers>=4.40` floor could not run this code, raised to `>=4.56` | adds `--config/--run_id/--seed`, `runlog.py`, `configs/r01.yaml`, `configs/r02.yaml`, `tests/test_runlog.py`. Unit checks 5/5 local; no end-to-end run yet (local env is transformers 4.40, no weights) |


## Per-run template (copy into notes/ or manifest)
- **runID / branch / commit:**
- **Command:** exact CLI line
- **Config:** path to YAML (J, h, cache size, max tokens, seed, dtype, subset definition)
- **Hardware:** GPU, driver, torch/transformers versions
- **Outputs:** paths under results/<runID>/
- **Deviations from plan:** anything that differed from plans/<branch>.md

## Failure log
Record every crash/OOM/env issue with fix. This section is evidence of debugging
process, not embarrassment — keep it.

| date | runID | symptom | root cause | fix |
|------|-------|---------|------------|-----|