# plans/q2-faithfulness.md

**Branch:** `q2-faithfulness` (from `main` @ a3682ae)
**runID:** Q2-01

## Hypothesis

The counter-causal surprise score claims to measure how predictable token x_i is from
future context alone. There is no oracle for the true backward conditional
P(x_i | x_{>i}), so faithfulness is tested as convergent validity: an independently
trained bidirectional MLM (ModernBERT), given only the future context, produces its own
estimate; if the flipped-mask scores rank tokens the way the referee does, the score
measures what it claims.

H1: per-sequence Spearman between the method's score and the referee's backward
    conditional is positive and remains so after controlling for word frequency, and is
    substantially closer to the forward-ceiling ρ than to zero.
H0: agreement is near zero, or vanishes once frequency is partialled out, or the score
    correlates with the referee's forward conditional as strongly as with its backward
    one (i.e. it captures generic predictability, not direction-specific information).

## Protocol

1. Sequences: `pred_raw` from results/R-00-qwen/raw (Qwen-generated MATH500 solutions,
   in-distribution for the scorer); first n_seqs=50 with 128–2048 Qwen tokens.
2. One causal Qwen forward per sequence harvests all-layer K/V, h^(L-1), and causal
   logits. The flipped passes then reuse those K/V exactly as the hooks do — never a
   from-scratch flipped forward, which would compute K/V from flipped states and test
   an algorithm nobody deploys.
3. Method scores per token, mirroring hooks.py: full (all layers flipped, fp32) and
   fast (last layer flipped, FFN skipped, native dtype). Both as raw logit (deployed)
   and log-softmax (distributional) variants.
4. Referee: ModernBERT-base. Backward conditional for word w = [CLS][MASK]×k + future
   tokens[SEP] (past dropped, not masked in place — long masked prefixes are OOD for
   MLM training); forward = [CLS] + past tokens + [MASK]×k[SEP]. Sum and mean of masked
   sub-token log-probs per word.
5. Word alignment: regex \S+ chunks of the raw text; both tokenizations mapped to char
   spans via offset mappings; a token belongs to a word if spans overlap. Words dropped
   if they contain Qwen token 0 or the last token, or have no tokens on either side.
   The same filtered word set feeds every vector.
6. Aggregation: mean over sub-tokens on both sides (primary); referee-sum and
   method-max as sensitivity. Mixed sum/mean would make word length a confound.
7. Controls: (a) forward ceiling — Spearman(Qwen causal log-prob, referee forward),
   the best agreement two trusted estimators achieve across this tokenizer gap;
   (b) frequency — corpus unigram -log f per word, reported raw and as partial
   Spearman; (c) direction — method score vs referee *forward* conditional.
8. Statistics: Spearman per sequence, distribution across sequences (mean, median,
   std); sequences with <30 scored words skipped.
9. Smoke: --n_seqs 2 before the full run. Deterministic throughout (seed recorded).
10. Isolation: fresh K/V deliberately excludes eviction-history staleness — that is
    Q3's variable, not Q2's.

## Expected runs

| runID | what | cost |
|-------|------|------|
| Q2-01 | 50 sequences, full pipeline | single GPU, ~30–60 min |

## Expected outputs

- results/Q2-01/metrics.json — headline: mean per-seq Spearman(full raw-logit, referee
  backward); all correlations and controls in detail
- results/Q2-01/raw/generations.jsonl.gz — per-sequence correlation records
- results/Q2-01/raw/words.csv.gz — per-word score table (plots for the meeting)
- Ledger row Q2-01 on this branch

## Interpretation guide (pre-registered)

- ρ_backward high relative to ceiling, survives frequency partial, ρ_forward low →
  score is faithful and direction-specific.
- ρ_backward ≈ ρ_forward → score measures generic predictability; the "counter-causal"
  framing is doing less work than claimed.
- ρ_backward ≈ 0 with intact ceiling → flipped-mask scores do not track a backward
  conditional; their eviction utility (R-11/Q1) rests on something else.
- Known mismatch, accepted: token-level scores see intra-word future (min_gap=0); the
  referee masks whole words. Direction of bias: inflates apparent method information.

## Out of scope

- Eviction-history staleness (Q3)
- min_gap > 0 variants
- LongHealth/Llama replication (only if the MATH500/Qwen result is ambiguous)
