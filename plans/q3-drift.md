# plans/q3-drift.md

**Branch:** `q3-drift` (from `main` @ 8883e74)
**runID:** Q3-01

## Hypothesis

Deployed counter-causal scoring reads K/V that are fossils: each surviving entry was
computed from whatever pruned past existed when its token was processed, and errors
compound across refresh cycles. Does score quality decay as fossils accumulate?

H1: practical-vs-oracle rank agreement declines with refresh cycle, and faster for
    counter-causal than for the Importance baseline (drift is the method's
    distinctive cost, motivating periodic recomputation).
H0a: both flat — drift is a non-issue for either signal (robustness result).
H0b: both decline in parallel — drift is real but method-agnostic.

Q2 linkage: scores are substantially frequency-shaped (Q2), and frequency does not
drift; flat lines are therefore a live outcome that would complete the story
coherently ("crude but stable").

## Design (per the pre-registered draft, with five refinements)

**Pass A (library).** Per conversation, chunked causal prefill with eviction OFF —
same chunk boundaries and kernel path as Pass B — harvesting every position's K/V at
every layer plus h^(L-1) to CPU. Refinement 1: same-loop construction makes the
caches bit-identical until the first eviction, so the cycle-1 sanity check is a hard
`torch.equal` on the cache, not a soft Spearman ≈ 1.

**Pass B (instrumented real run).** Chunked prefill, counter-causal (full) driving
eviction. At each refresh, before the argsort: log practical scores from fossil K/V
and oracle scores from library K/V gathered by original position (queries re-embedded
as usual), for three methods — CC-full, CC-fast (refinement 3: the deployed default;
oracle h^(L-1) from the library; measured on the full-driven trajectory), and
Importance on last-layer keys. Eviction then proceeds from CC-full practical scores,
untouched: the trajectory is never perturbed by the measurement.

**Why Importance and not H2O:** H2O's score is an accumulation over the whole
trajectory; a per-cycle fresh-K/V counterfactual is ill-defined for it. Importance is
stateless and oracle-izes exactly. (Its score is scale-invariant to the shared
normalizer, so the fossil/library normalizer difference cannot move its ranks.)

**What is deliberately NOT varied:** both practical and oracle score the same
surviving candidate set. Context truncation per se is not the manipulated variable —
only representation staleness is.

## Protocol

1. Llama-3.1-8B fp16 (deployment dtype), LoCoMo, J=4096, h=1024, greedy, seed 0.
2. Refinement 4: the 5 longest conversations by Llama token count (deterministic,
   maximizes cycles), prompt = system(actual speakers) + Input + conversation;
   frozen sinks = measured system-prompt tokens; prefill only, stop before decode.
3. Per cycle, over body entries minus frozen sinks and the pinned last entry:
   Spearman(practical, oracle) per method.
4. Refinement 2: decision metric = Jaccard of EVICTED sets (chance ≈ 0.11), since
   keep-set Jaccard is pigeonhole-bounded ≥ 0.60 at J/n = 4096/5120 and hides drift.
   Keep-set Jaccard logged too.
5. Refinement 5: mechanism metric — mean cosine(fossil K, library K), layers 16 and
   31, per cycle. Note (amended 2026-08-10, pre-run): each K is computed once and
   never updated, so per-position cosine is fixed at birth; the survivor mean moves
   only through population turnover. Therefore ALSO logged per cycle: the newest
   cohort's mean cosine — if it declines with cycle index, distortion compounds
   recursively; if flat, each cohort is equally distorted and drift is bounded.
6. Sanity: cycle-1 cache equality assert; cycle-1 Spearman must be 1.0.
7. Outputs: per-cycle jsonl + metrics.json summary + scripts/plot_q3.py figure
   (cycle index vs Spearman / evicted-Jaccard, per-method mean with IQR band across
   conversations; per-cycle support shrinks at high cycles as conversations end).

## Expected runs

| runID | what | cost |
|-------|------|------|
| Q3-01 | 5 conversations, full instrumentation | ~15-20 cycles/conv, minutes each |

## Out of scope

- Decode-phase drift (counterfactual not computable once eviction alters generation —
  the reason this design is prefill-only).
- A fast-driven eviction trajectory (noted caveat; second trajectory doubles cost).
- H2O (above). Downstream task impact of drift (this measures the score, not QA).
