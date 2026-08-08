# plans/q1-layer-sweep.md

**Branch:** `q1-layer-sweep` (from `main` @ ea44d71)
**runIDs:** Q1-01, Q1-02, Q1-03

## Hypothesis

The two upstream hooks are endpoints of one family. Write ℓ for the first layer whose
attention uses the flipped (counter-causal) mask during scoring:

- counter-causal (full)  = ℓ = 0   (every layer flipped)
- counter-causal (fast)  ≈ ℓ = 31  (only the last layer flipped, plus two extra
  approximations: the FFN of that layer is skipped, and h^(30) is reused from the
  causal forward instead of recomputed)

If the eviction signal is created mostly in the top layers, quality should hold as ℓ
rises from 0 toward 31 and then drop somewhere; where it drops locates the layers that
matter. H1: there is an ℓ ≥ 16 whose accuracy matches ℓ=0 (R-11-full) within noise.
H0: accuracy degrades monotonically from ℓ=0, i.e. every layer's flipped pass contributes.

**Supersedes the CLAUDE.md Q1 sketch.** The original sketch proposed logit-lens scoring
from a single layer ℓ. This protocol instead varies where the mask flips inside one
full-depth scoring pass — layers 0..ℓ-1 causal, ℓ..31 counter-causal — which isolates
the mask position from the depth-truncation and FFN-skip approximations. Decision made
2026-08-09 before any Q1 code.

## Protocol

1. New module `q1_hooks.py`: `counter_causal_split_hook(model, flip_from_layer=ℓ, ...)`,
   a copy of the full hook's layer walk where layer_idx < ℓ uses the ordinary causal
   mask (tril, self included) and layer_idx ≥ ℓ uses the flipped mask (triu, 1+min_gap).
   Cached K/V reused at every layer; Q fresh; FFN applied normally; fp32 logits.
2. `evaluate.py`: hook choice `counter_split` + `--flip_from_layer` (required with it).
3. Sanity identity: ℓ=0 must equal `counter_causal_hook` exactly (same keep-set).
4. Three runs on LongHealth (all 400), Llama-3.1-8B, fp16, cache 8000, auto_frozen,
   seed 0 — identical to the R-11 group except the hook.
5. Compare against R-11-full (ℓ=0, acc in ledger) and R-11-fast (≈ℓ=31).

## Expected runs

| runID | flip_from_layer | causal layers | flipped layers | config |
|-------|-----------------|---------------|----------------|--------|
| Q1-01 | 8  | 0–7  | 8–31  | configs/q1_flip8.yaml |
| Q1-02 | 16 | 0–15 | 16–31 | configs/q1_flip16.yaml |
| Q1-03 | 24 | 0–23 | 24–31 | configs/q1_flip24.yaml |

Cost note: the split hook recomputes all 32 layers whatever ℓ is, so these runs cost
what R-11-full cost — the sweep locates the signal, it is not itself a speedup. If some
ℓ matches full quality, the natural follow-up is a true fast-ℓ variant (capture h^(ℓ-1),
run only layers ℓ..31 flipped), whose cost scales with (32-ℓ)/32.

Comparison caveat: split(31) is not identical to counter-causal (fast) — split applies
the last FFN, recomputes the prefix from the current cache, and scores in fp32. The
clean anchors are ℓ=0 (exact) and the R-11-fast number (approximate upper-ℓ reference).

## Expected outputs

- `results/Q1-0{1..3}/` with metrics.json, manifest.json, raw dump
- Ledger rows Q1-01..03 on this branch (pre-registered as queued)
- Verdict in the report: accuracy vs ℓ curve with R-11 endpoints

## Out of scope

- The capture-based fast-ℓ variant (follow-up plan if the sweep motivates it)
- Any change to `hooks.py`, `tasks.py`, or main
- Qwen2.5 replication of the sweep (only if the Llama curve is interesting)
