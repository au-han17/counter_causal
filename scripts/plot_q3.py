"""
Figure for Q3: practical-vs-oracle agreement per refresh cycle.

Two panels — Spearman and evicted-set Jaccard — three methods (cc, fast, imp),
per-cycle mean across conversations with an interquartile band. The evicted-Jaccard
chance level (~0.11 at J=4096, n=5120) is drawn as a dashed floor.

    python scripts/plot_q3.py [results/Q3-01] [--out results/Q3-01/figure.png]
"""

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="results/Q3-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed: pip install matplotlib")

    import numpy as np

    path = os.path.join(args.run_dir, "raw", "generations.jsonl.gz")
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8")]
    out = args.out or os.path.join(args.run_dir, "figure.png")

    by_cycle = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                by_cycle[r["cycle"]][k].append(v)
    cycles = sorted(by_cycle)

    def series(key):
        med, lo, hi = [], [], []
        for c in cycles:
            v = np.array(by_cycle[c][key])
            med.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        return np.array(med), np.array(lo), np.array(hi)

    # chance level for evicted-set jaccard: E|A∩B| = e²/n for independent size-e sets
    with open(os.path.join(args.run_dir, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    e = rows[0]["n_body"] - m["cache_size"]
    exp_inter = e * e / rows[0]["n_body"]
    chance = exp_inter / (2 * e - exp_inter)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    colors = {"cc": "#c0392b", "fast": "#2471a3", "imp": "#7d6608"}
    labels = {"cc": "counter-causal (full)", "fast": "counter-causal (fast)",
              "imp": "importance"}

    for m_ in ("cc", "fast", "imp"):
        med, lo, hi = series(f"spearman_{m_}")
        axes[0].plot(cycles, med, color=colors[m_], label=labels[m_])
        axes[0].fill_between(cycles, lo, hi, color=colors[m_], alpha=0.18, lw=0)
        med, lo, hi = series(f"jac_evict_{m_}")
        axes[1].plot(cycles, med, color=colors[m_], label=labels[m_])
        axes[1].fill_between(cycles, lo, hi, color=colors[m_], alpha=0.18, lw=0)

    axes[0].set_ylabel("Spearman(practical, oracle)")
    axes[0].set_ylim(top=1.02)
    axes[1].set_ylabel("evicted-set Jaccard (practical vs oracle)")
    axes[1].axhline(chance, ls="--", c="gray", lw=1)
    axes[1].text(cycles[-1], chance, " chance", va="center", fontsize=8, color="gray")
    for ax in axes:
        ax.set_xlabel("refresh cycle")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Score drift under fossil K/V (median across conversations, IQR band)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"written {out}")


if __name__ == "__main__":
    main()
