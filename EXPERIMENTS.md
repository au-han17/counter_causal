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

## Ledger

| runID | date | branch | commit | model | dataset (slice) | strategy | config | GPU-h | status | headline result | notes |
|-------|------|--------|--------|-------|-----------------|----------|--------|-------|--------|-----------------|-------|
| R-01 | | main | | Llama-3.1-8B | MATH500 | full | configs/r01.yaml | | | | env sanity anchor; target ≈.488 |
| R-02 | | main | | Llama-3.1-8B | MATH500 | sliding | configs/r02.yaml | | | | target ≈.458 |
| R-03 | | main | | Llama-3.1-8B | MATH500 | H2O | | | | | target ≈.464 |
| R-04 | | main | | Llama-3.1-8B | MATH500 | counter-causal (full) | | | | | target ≈.482; enable --dump_scores |
| R-05 | | main | | Llama-3.1-8B | MATH500 | counter-causal (fast) | | | | | target ≈.480 |
| R-06..R-10 | | main | | Qwen2.5-7B | MATH500 | same five | | | | | targets .766/.692/.762/.744/.736 |
| R-11 | | main | | Llama-3.1-8B | LongHealth (all 400) | counter-causal (full) vs (fast) | anomaly cache size 8000 | | | | anomaly: counter-causal fast > counter-causal full? |
| R-12 | | main | | Llama-3.1-8B | LongHealth | same | anomaly cache size 9000 | | | | robustness of anomaly |
| R-13 | | main | | Llama-3.1-8B | LoCoMo multi-hop (282) | five strategies | anomaly cache size 15000 | | | | anomaly: fast > full-cache? H2O collapse? |
| R-14 | | main | | Llama-3.1-8B | LoCoMo multi-hop (282) | five strategies | adjacent cache size 17000 | | | | robustness |


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