"""
Paired analysis of the LongHealth runs: R-11 group + Q1 layer sweep.

Every run answered the same 400 questions, so paired tests are far more powerful
than comparing raw accuracies (SE of a single accuracy at p~0.8, n=400 is ~2 points;
McNemar on the discordant pairs resolves much smaller gaps when runs mostly agree).

Reads results/<runID>/raw/generations.jsonl.gz, keys samples by
(patient_id, question_no), recomputes accuracy as an integrity check against
metrics.json, then reports exact two-sided McNemar for every pair.

Usage:
    python scripts/analyze_q1.py
    python scripts/analyze_q1.py --runs R-11-fast R-11-h2o
    python scripts/analyze_q1.py --out report/q1_paired_analysis.md
"""

import argparse
import gzip
import io
import json
import math
import os
import sys

# (runID, short label) in presentation order
DEFAULT_RUNS = [
    ("R-11-none", "no eviction (ceiling)"),
    ("R-11-full", "counter-causal full (flip 0)"),
    ("Q1-01",     "split (flip 8)"),
    ("Q1-02",     "split (flip 16)"),
    ("Q1-03",     "split (flip 24)"),
    ("R-11-fast", "counter-causal fast (last layer)"),
    ("R-11-h2o",  "H2O"),
]


def load_run(results_root, run_id):
    """Return {(patient_id, question_no): bool_correct} or None if missing."""
    path = os.path.join(results_root, run_id, "raw", "generations.jsonl.gz")
    if not os.path.exists(path):
        return None
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["patient_id"], r["question_no"])] = bool(r["correct"])
    return out


def metrics_score(results_root, run_id):
    path = os.path.join(results_root, run_id, "metrics.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f).get("score")


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value (binomial sign test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main():
    ap = argparse.ArgumentParser(description="Paired McNemar analysis of LongHealth runs")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="runIDs to compare (default: the R-11 group + Q1 sweep)")
    ap.add_argument("--out", default=None, help="also write the report to this markdown file")
    args = ap.parse_args()

    wanted = [(r, r) for r in args.runs] if args.runs else DEFAULT_RUNS

    runs = {}     # run_id -> {key: correct}
    labels = {}
    for run_id, label in wanted:
        data = load_run(args.results_root, run_id)
        if data is None:
            print(f"warn: {run_id} has no raw dump under {args.results_root}/ — skipped")
            continue
        runs[run_id] = data
        labels[run_id] = label
    if len(runs) < 2:
        sys.exit("need at least two runs with raw dumps")

    ids = list(runs)
    common = set.intersection(*(set(v) for v in runs.values()))
    union = set.union(*(set(v) for v in runs.values()))
    if len(common) < len(union):
        print(f"warn: aligning on {len(common)} common questions "
              f"(union {len(union)}; some runs miss samples)")

    keys = sorted(common)
    lines = []
    say = lines.append

    say("## Per-run accuracy (recomputed from raw dumps)")
    say("")
    say("| run | label | n | accuracy | metrics.json |")
    say("|---|---|---|---|---|")
    for rid in ids:
        acc = sum(runs[rid][k] for k in keys) / len(keys)
        rec = metrics_score(args.results_root, rid)
        tag = "ok" if rec is not None and abs(rec - acc) < 5e-4 else ("MISMATCH" if rec is not None else "-")
        say(f"| {rid} | {labels[rid]} | {len(keys)} | {acc:.4f} | {rec if rec is not None else '-'} {tag} |")
    say("")

    n_all_right = sum(all(runs[r][k] for r in ids) for k in keys)
    n_all_wrong = sum(not any(runs[r][k] for r in ids) for k in keys)
    say("## Question structure")
    say("")
    say(f"- every run correct: {n_all_right}")
    say(f"- every run wrong:   {n_all_wrong}")
    say(f"- contested (at least one run differs): {len(keys) - n_all_right - n_all_wrong}")
    say("")

    say("## Pairwise McNemar (exact, two-sided)")
    say("")
    say("| A | B | acc A | acc B | both right | both wrong | A only | B only | agree | p |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    n_pairs = 0
    for i, a in enumerate(ids):
        for bb in ids[i + 1:]:
            ra, rb = runs[a], runs[bb]
            both_r = sum(ra[k] and rb[k] for k in keys)
            both_w = sum(not ra[k] and not rb[k] for k in keys)
            a_only = sum(ra[k] and not rb[k] for k in keys)
            b_only = sum(rb[k] and not ra[k] for k in keys)
            p = mcnemar_exact(a_only, b_only)
            agree = (both_r + both_w) / len(keys)
            acc_a = (both_r + a_only) / len(keys)
            acc_b = (both_r + b_only) / len(keys)
            mark = " *" if p < 0.05 else ""
            say(f"| {a} | {bb} | {acc_a:.3f} | {acc_b:.3f} | {both_r} | {both_w} "
                f"| {a_only} | {b_only} | {agree:.1%} | {p:.4f}{mark} |")
            n_pairs += 1
    say("")
    say(f"`*` p < 0.05 uncorrected. {n_pairs} comparisons — "
        f"Bonferroni-adjusted threshold ~{0.05 / n_pairs:.4f}.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with io.open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(report + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
