"""
Close the Q2 analytic gap: frequency-partial Spearman for the LOG-PROB score variants.

q2_faithfulness.py computed the frequency partial only for the deployed raw-logit
scores. The log-prob variant is the stronger backward correlate (+0.148 vs +0.097),
so whether IT survives the frequency control decides between:
  - "all faithfulness is frequency" (partial ~0), and
  - "log-prob carries genuine backward signal the deployed raw-logit score wastes".

Reads results/Q2-01/raw/words.csv.gz. No GPU, runs in seconds.

    python scripts/q2_logprob_partial.py [results/Q2-01/raw/words.csv.gz]
"""

import gzip
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from q2_faithfulness import partial_spearman, spearman  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/Q2-01/raw/words.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        n_numeric = len(header) - 1  # 'word' is first; it may contain commas
        rows = []
        for line in f:
            parts = line.rstrip("\n").split(",")
            word = ",".join(parts[:len(parts) - n_numeric])
            vals = dict(zip(header[1:], parts[len(parts) - n_numeric:]))
            vals["word"] = word
            rows.append(vals)

    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq_id"]].append(r)

    stats = defaultdict(list)
    for seq, rs in by_seq.items():
        if len(rs) < 30:
            continue
        col = lambda k: [float(r[k]) for r in rs]
        stats["partial_full_logprob_bwd_freq"].append(
            partial_spearman(col("full_logprob"), col("ref_bwd"), col("neglogfreq")))
        stats["partial_fast_logprob_bwd_freq"].append(
            partial_spearman(col("fast_logprob"), col("ref_bwd"), col("neglogfreq")))
        stats["rho_full_logprob_freq"].append(
            spearman(col("full_logprob"), col("neglogfreq")))

    print(f"{len(by_seq)} sequences, {len(rows)} words")
    for k, vals in stats.items():
        vals = [v for v in vals if v == v]  # drop nan
        vals.sort()
        mean = sum(vals) / len(vals)
        median = vals[len(vals) // 2]
        pos = sum(v > 0 for v in vals) / len(vals)
        print(f"  {k:34} mean {mean:+.4f}  median {median:+.4f}  "
              f"{pos:.0%} positive  (n={len(vals)})")


if __name__ == "__main__":
    main()
