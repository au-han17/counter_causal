# plans/dtype-verification.md

**Branch:** `dtype-verification` (from `main` @ 2c8241b)
**runID:** `R-00` — preliminary finding, sits before the R-01..R-14 reproduction runs

## Hypothesis

`evaluate.py` hardcoded `dtype=torch.float16`, but both target checkpoints ship in
bfloat16 (verified from HF `config.json`: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct
both declare `torch_dtype: bfloat16`). fp16 keeps a 5-bit exponent (max ≈65504) against
bf16's 8-bit, so the downcast trades range for mantissa and can overflow on activations
that were in-range during training. The counter-causal hooks attend over the whole
cached sequence at once and wrap the result in `torch.nan_to_num`
(`hooks.py:350`, `hooks.py:446`), so any overflow is silently zeroed rather than raised.

H1: loading in bf16 moves MATH500 accuracy closer to the paper's reference numbers than
    the fp16 default.
H0: the difference is within run-to-run noise and fp16 is a harmless deviation.

**Secondary interaction (not tested here, informs Q-phase work).** Scoring dtype differs
between the two hooks: `counter_causal_hook` casts logits to fp32 (`hooks.py:357`),
`counter_causal_fast_hook` does not (`hooks.py:451`). Under fp16 weights the fast hook
ranks in fp16 (10 mantissa bits); under bf16 weights it ranks in bf16 (7 mantissa bits).
So switching weights to bf16 makes the fast hook's ranking *coarser*, and any accuracy
change for `counter_fast` confounds two effects. R-05 and R-10 are therefore read as
confounded until scoring dtype is addressed separately. Scoring dtype is left untouched
on this branch.

## Protocol

1. Add `--dtype {float16,bfloat16,float32}` to `evaluate.py`; default `float16` so
   existing behaviour is byte-identical unless the flag is passed.
2. Thread it into `from_pretrained` and record the resolved value in the output JSON.
3. Smoke test: `--single_sample 0` on MATH500, both dtypes, confirm the flag takes effect.
4. Run the reproduction rows below with `--dtype bfloat16`, MATH500, full 500 problems,
   greedy, single seed.
5. Compare against paper reference, and against the same row run in fp16.
6. If |Δ| > 0.02 on any pair, write a follow-up plan before changing scoring dtype.

## Expected runs

Two runs, both under R-00, executed from the integration branch `r00-bf16`
(= `run-harness` + `dtype-verification`, so each run writes a manifest):

| runID | model | strategy | dtype | paper target | config |
|-------|-------|----------|-------|--------------|--------|
| R-00-qwen | Qwen2.5-7B-Instruct | full (no eviction) | bfloat16 | .766 | configs/r00_qwen.yaml |
| R-00-llama | Llama-3.1-8B-Instruct | full (no eviction) | bfloat16 | .488 | configs/r00_llama.yaml |

`hook: none` in both, so weight dtype is isolated from any eviction or scoring-dtype
effect. The fp16 comparison points are the upstream author's existing numbers; a matched
fp16 rerun can be added later by passing `--dtype float16` with the same config.

## Expected outputs

- `results/R-00/manifest.json` — config path, git commit, GPU, torch/transformers
  versions; no metrics.json, since no evaluation is run
- `dtype` field present in every downstream run's output JSON
- One EXPERIMENTS.md ledger row for R-00
- Verdict recorded in the Preliminary Precision Verification section: adopt bf16 as the
  reproduction default, or document fp16 as acceptable

## Out of scope

- Scoring-dtype fix at `hooks.py:451` (pending R-01/R-06 results)
- Timing/memory instrumentation (no wall-time capture exists yet; would need its own plan)
- Any change to eviction or scoring logic
