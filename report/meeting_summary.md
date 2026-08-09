# Counter-Causal KV-Cache Eviction — Reproduction & Analysis

**Meeting summary — 2026-08-10.**
Repo: `au-han17/counter_causal` (fork of upstream). Every run has a pre-registered
plan (`plans/`), a config (`configs/`), a manifest with commit/seed/env
(`results/<runID>/manifest.json`), and a ledger row (`EXPERIMENTS.md`). Branches:
`main` (reproduction), `q1-layer-sweep`, `q2-faithfulness`, `q3-drift`.

## Executive summary

1. **The method reproduces.** MATH500/Qwen numbers land within ~6 points of a 2-point
   SE of the paper's Table 2; on LongHealth the fast variant is statistically
   indistinguishable from no eviction at 1/11th the scoring cost of the full variant.
2. **The mechanism doesn't need depth (Q1).** Flipping the mask at layer 0, 8, 16, 24
   or 31 produces statistically indistinguishable eviction quality.
3. **The mechanism is not what the paper says it is (Q2).** Tested against an external
   backward-conditional referee, the full score is weakly faithful — and most of that
   is word frequency. The deployed fast score does not track a backward conditional at
   all. Two scores that agree with each other at only ρ = 0.32 evict equally well.
4. **Fossil-K/V drift is real, monotone, and ~3× worse for counter-causal than for a
   shallow baseline — but small (Q3).** Practical-vs-oracle rank agreement stays ≥ .975
   through 11+ refresh cycles; ~10% of CC's per-cycle eviction decisions flip vs ~3%
   for Importance, against representations that barely move (cos ≥ .99). The deep
   scoring pass amplifies representation noise; the damage stays well inside margins
   that Q1 showed do not affect accuracy.

**Synthesis:** the eviction utility of counter-causal scoring is not explained by
backward-conditional faithfulness. Eviction succeeds because "keep rare/surprising,
drop redundant" is a low bar many scoring functions clear — consistent with the Q1
depth-invariance, with only 47/400 LongHealth questions being contested, and with
Q3's finding that even a 10%-per-cycle decision perturbation is far smaller than
method-to-method differences that already left accuracy unchanged.

---

## 1. Reproduction (scope-reduced by design; deferred rows remain in the ledger)

MATH500, Qwen2.5-7B-Instruct, J=512, h=256, greedy, seed 0:

| run | strategy | dtype | ours | paper |
|---|---|---|---|---|
| R-06 | full cache | fp16 | .760 | .766 |
| R-00-qwen | full cache | bf16 | .764 | (.766) |
| R-09 | counter-causal (full) | fp16 | .740 | .744 |
| R-10 | counter-causal (fast) | fp16 | .738 | .736 |

All within noise of the paper (SE ≈ 1.9 pts at n=500). **Side finding:** upstream
loads both natively-bf16 checkpoints in fp16; R-00-qwen vs R-06 shows the downcast is
benign here (.764 vs .760).

LongHealth, Llama-3.1-8B, cache 8000, frozen system prompt, seed 0 (n=400):

| strategy | accuracy | hook cost (2445 calls) |
|---|---|---|
| no eviction (ceiling) | .810 | — |
| counter-causal (fast) | .805 | 73 s |
| counter-causal (full) | .790 | 821 s |
| H2O (same budget) | .775 | — |

## 2. Q1 — does the flip depth matter? No.

Split hook: layers 0..ℓ-1 causal, ℓ..31 flipped (ℓ=0 ≡ full; ℓ≈31 ≈ fast).
Accuracy: flip8 .7975, flip16 .7925, flip24 .795 — between full (.790) and fast (.805).

Paired McNemar over the same 400 questions (`report/q1_paired_analysis.md`):

- **fast vs no-eviction: p = 0.79** (8 vs 6 discordant) — statistically at the ceiling.
- **no-eviction vs H2O: p = 0.0066** (19 vs 5) — the only sub-.05 comparison of 21;
  fast vs H2O p = 0.0501 (22 vs 10) points the same way. H2O is the only method with
  evidence of real damage. (Bonferroni threshold 0.0024 — "clearest signal", not
  "proven beyond correction".)
- **Everything among {full, flip8, flip16, flip24, fast}: p ≥ 0.26.** Full vs flip8
  agree on 98.2% of questions (2 vs 5 discordant). Depth of the flipped region is
  irrelevant to eviction quality.
- Benchmark caveat: 290/400 questions solved by every method, 63 by none — the whole
  comparison lives in 47 questions. LongHealth at this budget has limited
  discriminative power, which is itself part of the story.

## 3. Q2 — is the surprise score a backward conditional? Barely / no.

Design (pre-registered in `plans/q2-faithfulness.md`, controls fixed before running):
scores recomputed on fresh K/V exactly as the hooks compute them (causal forward
harvests K/V; flipped pass on top); referee = ModernBERT cloze given only future
context; word-level alignment across tokenizers via character spans; per-sequence
Spearman, n=50 Qwen-generated MATH500 solutions.

| quantity | mean ρ | % seqs positive |
|---|---|---|
| **ceiling**: Qwen forward vs referee forward | **+0.348** | 100% |
| full (raw logit, deployed) vs referee backward | +0.097 | 90% |
| full (log-prob) vs referee backward | +0.148 | 94% |
| fast (raw logit, deployed) vs referee backward | +0.015 | 58% |
| fast (log-prob) vs referee backward | −0.034 | 44% |
| full vs referee **forward** (direction control) | +0.036 | 66% |
| full vs backward, **frequency partialled out** | +0.039 | 70% |
| full (log-prob) vs backward, frequency partialled out | **+0.072** | 82% |
| fast vs backward, frequency partialled out | −0.045 | 36% |
| fast (log-prob) vs backward, frequency partialled out | −0.072 | 32% |
| full score vs fast score | +0.324 | 98% |

Reading (per the pre-registered interpretation guide):

- **Full:** faithful but weak — 28% of ceiling as deployed, 42% with log-probs;
  direction-specific (backward ≫ forward); **~60% of the deployed signal is word
  frequency** (0.097 → 0.039 after partial; residual real, sign test p ≈ 0.007, but
  small). The log-prob variant's residual is roughly **double**: +0.072 after the
  frequency partial, positive in 82% of sequences (sign test p ≈ 6×10⁻⁶). Log-prob
  scoring carries genuine backward signal that the deployed raw-logit score halves.
- **Fast — the deployed default:** does not track a backward conditional. ~Zero
  against the referee in both directions, negative after the frequency partial in
  both score variants (raw −0.045, log-prob −0.072; 36%/32% of sequences positive).
- Yet fast evicts at the ceiling (Q1) and full-fast rank agreement is only 0.32.

## 4. Q3 — does fossil K/V drift corrupt the score? Measurably; not consequentially.

Design (pre-registered in `plans/q3-drift.md`): per conversation, a drift-free
**library** (eviction-free chunked prefill, same chunk boundaries and kernel path,
all-layer K/V to CPU) and an **instrumented run** (cc-full-driven eviction, J=4096,
h=1024, 5 longest LoCoMo conversations, prefill only). At each refresh cycle,
practical scores from the fossil cache vs oracle scores from library K/V gathered by
original position — for CC-full and for Importance on the same fossils — then
eviction proceeds from the practical scores, untouched. Built-in validation: until
the first eviction the two caches are bit-identical; cycle-1 agreement was exactly
1.0 everywhere.

| cycle band | CC Spearman | Imp Spearman | | evicted-set agreement (chance ≈ 11%) |
|---|---|---|---|---|
| 1 (sanity) | 1.000 | 1.000 | | CC ≈ 90% of 1024 choices match oracle |
| 2–5 | .991 | .996 | | Imp ≈ 97% |
| 6–10 | .985 | .992 | | |
| 11+ | .975 | .988 | | |

- **Drift is real and monotone** — every band declines, no plateau within the
  measured horizon.
- **It is method-tilted ≈ 2–3×:** CC loses 2.1× more rank correlation and flips ~3×
  more eviction decisions (~10% vs ~3.4% per cycle) than Importance *on the same
  fossil cache, same trajectory*.
- **Mechanism:** representations barely move — fossil-vs-library cosine ≥ .99 at
  layers 16 and 31, with new cohorts only slightly worse (.988/.994). The 32-layer
  flipped scoring pass amplifies that whisper of noise into decision changes; the
  one-layer Importance dot-product does not. Drift contaminates the deep measurement
  instrument, not the cache itself.
- **Bounded by Q1:** practical-vs-oracle eviction agreement (~90%) is far tighter
  than the fast-vs-full score agreement (ρ ≈ .32) that already produced identical
  accuracy. Downstream impact at these settings is therefore expected to be nil; the
  paper's periodic-refresh design is adequate without modification.

## Implications

- The paper's *empirical* claims hold up. The *explanatory* claim — that eviction
  works because the score measures predictability-from-the-future — is not supported:
  the variant that evicts best measures it least.
- Practical consequences: (a) the fast variant is strictly preferable at this budget —
  cheaper and never worse; (b) scoring with log-probs instead of raw logits is a
  one-line change that doubles the frequency-adjusted backward signal
  (+0.039 → +0.072), eviction impact untested;
  (c) benchmarks with more contested mass are needed to separate scoring functions
  at all.
- **Q3 answered the staleness question in the direction Q2 predicted:** scores whose
  utility is frequency-shaped rather than context-shaped should tolerate fossil K/V —
  and they do. Drift is measurable and method-tilted, not consequential. The paper's
  periodic-refresh scheme needs no repair at these settings.

## Limitations (stated, not hidden)

- LongHealth n=400 resolves ~2-pt gaps only via pairing; single seed, greedy decoding.
- Reproduction runs executed concurrently on one GPU: accuracies are exact,
  wall-clock/peak-memory fields are contended (hook-time ratios are per-call sums and
  remain valid).
- Referee is an MLM on LaTeX-dense text — the 0.348 ceiling caps all comparisons;
  ratios-to-ceiling are the honest unit.
- Token-level scores see intra-word future (min_gap=0); referee masks whole words —
  biases *toward* finding method information.
- Q2 scored bf16; deployed R-runs are fp16. Q2 text is MATH500 only.
- Q3 is prefill-only (the no-eviction counterfactual is not computable once eviction
  alters generation) and single-trajectory (both methods measured on cc-full-driven
  fossils; an Importance-driven run would fossilize differently). Decision flips are
  per-cycle vs oracle; trajectory-level compounding beyond ~20 cycles unmeasured.
- All **DIRTY TREE** ledger flags before 2026-08-09 are a false positive (the run's
  own untracked output directory tripped the check — since fixed); every run executed
  from a clean tracked tree at its recorded commit.

## Upstream fixes found during reproduction

- `requirements.txt` floor `transformers>=4.40` cannot run this code (needs
  `DynamicCache.layers`, 4.54; `from_pretrained(dtype=)`, 4.56). Raised to 4.56.
- Both checkpoints are natively bf16, loaded as fp16 (benign here; now a flag).
- README's LoCoMo example labels category 1 "single-hop"; the dataset shows category 1
  is multi-hop (282 QA, mean 3.13 evidence items, 98% ≥ 2) — the code scores it
  correctly; the label is wrong.

## Pointers

`EXPERIMENTS.md` (ledger) · `plans/` (pre-registered designs) ·
`report/q1_paired_analysis.md` (full McNemar table) ·
`results/Q3-01/figure.png` (drift curves, via `scripts/plot_q3.py`) ·
`results/<runID>/` (metrics + manifests + raw dumps) ·
branches `q1-layer-sweep`, `q2-faithfulness`, `q3-drift`
