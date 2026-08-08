"""
Fill an EXPERIMENTS.md ledger row from a completed run's artefacts.

    python scripts/ledger_update.py R-01
    python scripts/ledger_update.py R-01 --failed "OOM during prefill"

Updates the row whose first cell is exactly <runID>; appends a new row if none
exists (the ledger pre-registers R-06..R-10 as one row, and R-11..R-14 without the
per-strategy suffix). Append-only in spirit: existing rows are filled, never removed.
"""

import argparse
import io
import json
import os
import re
import sys
from datetime import datetime, timezone

LEDGER = "EXPERIMENTS.md"
COLUMNS = 12  # runID|date|branch|commit|model|dataset|strategy|config|GPU-h|status|headline|notes


def _read_json(path):
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def _cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def build_cells(run_id, metrics, manifest, failed_reason):
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    git = (manifest or {}).get("git") or {}
    env = (manifest or {}).get("env") or {}
    timing = (metrics or {}).get("timing") or (manifest or {}).get("timing") or {}

    wall = timing.get("wall_sec")
    gpu_h = f"{wall / 3600:.2f}" if isinstance(wall, (int, float)) else ""

    if failed_reason:
        status, headline = "FAILED", ""
    else:
        status = "done"
        score = (metrics or {}).get("score")
        key = (metrics or {}).get("score_key", "score")
        headline = f"{key}={score:.4f}" if isinstance(score, (int, float)) else ""

    notes = []
    if metrics:
        for k in ("dtype", "hook", "cache_size", "chunk_size", "frozen_size"):
            if metrics.get(k) not in (None, ""):
                notes.append(f"{k}={metrics[k]}")
    if timing.get("peak_gpu_gb"):
        notes.append(f"peak={timing['peak_gpu_gb']}GB")
    if timing.get("hook_sec") is not None:
        notes.append(f"hook={timing['hook_sec']}s/{timing.get('hook_calls')}calls")
    if env.get("transformers"):
        notes.append(f"tf={env['transformers']}")
    if git.get("dirty"):
        notes.append("**DIRTY TREE**")
    if failed_reason:
        notes.append(failed_reason)

    return {
        "runID": run_id,
        "date": date,
        "branch": git.get("branch") or "main",
        "commit": git.get("commit_short") or "",
        "model": (metrics or {}).get("model", "").split("/")[-1],
        "dataset": _slice_str(manifest),
        "strategy": (metrics or {}).get("hook", ""),
        "config": (manifest or {}).get("config_path") or "",
        "gpu_h": gpu_h,
        "status": status,
        "headline": headline,
        "notes": "; ".join(notes),
    }


def _slice_str(manifest):
    sl = (manifest or {}).get("dataset_slice") or {}
    parts = [sl.get("task", "")]
    if sl.get("n_samples") is not None:
        parts.append(f"({sl['n_samples']})")
    for k in ("category", "level", "subject"):
        if sl.get(k) is not None:
            parts.append(f"{k}={sl[k]}")
    return " ".join(p for p in parts if p)


def apply(run_id, cells, ledger=LEDGER):
    with io.open(ledger, encoding="utf-8") as f:
        lines = f.read().split("\n")

    ordered = [cells[k] for k in ("runID", "date", "branch", "commit", "model", "dataset",
                                 "strategy", "config", "gpu_h", "status", "headline", "notes")]

    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        c = _cells(line)
        if len(c) == COLUMNS and c[0] == run_id:
            # Preserve any hand-written notes by appending, not replacing.
            if c[11] and c[11] not in ordered[11]:
                ordered[11] = f"{ordered[11]}; {c[11]}" if ordered[11] else c[11]
            lines[i] = _row(ordered)
            _write(ledger, lines)
            return "updated"

    last = max(i for i, l in enumerate(lines)
               if l.startswith("|") and len(_cells(l)) == COLUMNS)
    lines.insert(last + 1, _row(ordered))
    _write(ledger, lines)
    return "appended"


def log_failure(run_id, reason, ledger=LEDGER):
    """Add a row to the Failure log table at the end of the ledger."""
    with io.open(ledger, encoding="utf-8") as f:
        text = f.read()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = f"\n| {date} | {run_id} | {reason} | (to investigate) | (pending) |"
    if not text.rstrip().endswith("|"):
        text = text.rstrip() + "\n"
    text = text.rstrip() + entry + "\n"
    with io.open(ledger, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write(path, lines):
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Fill an EXPERIMENTS.md row from run artefacts")
    ap.add_argument("run_id")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--failed", default=None, help="Mark the run FAILED with this reason")
    args = ap.parse_args()

    d = os.path.join(args.results_root, args.run_id)
    metrics = _read_json(os.path.join(d, "metrics.json"))
    manifest = _read_json(os.path.join(d, "manifest.json"))

    if metrics is None and not args.failed:
        sys.exit(f"{d}/metrics.json missing — run did not complete; "
                 f"pass --failed <reason> to record it as a failure")

    cells = build_cells(args.run_id, metrics, manifest, args.failed)
    action = apply(args.run_id, cells)
    if args.failed:
        log_failure(args.run_id, args.failed)
    print(f"{action} ledger row for {args.run_id}: "
          f"status={cells['status']} {cells['headline']} gpu_h={cells['gpu_h']}")


if __name__ == "__main__":
    main()
