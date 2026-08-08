"""
Run bookkeeping for the counter-causal reproduction: config loading, environment
capture, timing, and manifest writing.

Kept separate from evaluate.py so that hooks.py and tasks.py stay untouched.

Public API
----------
load_config              — read a run YAML into a dict of argparse defaults
seed_everything          — seed python/numpy/torch, return the seed
TimedHook                — wrap a kv_hook to accumulate GPU-synchronised call time
RunRecorder              — wall clock, peak GPU memory, and manifest/metrics writing
"""

import gzip
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from time import perf_counter

import torch

_MAX_RAW_MB = 50  # CLAUDE.md: commit raw generations only below this


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str, valid_keys) -> dict:
    """
    Load a run YAML into a dict suitable for argparse's set_defaults.

    Unknown keys raise rather than silently no-op — a typo in a config must not
    produce a run that looks configured but isn't.
    """
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("--config requires pyyaml (pip install pyyaml)") from e

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top level must be a mapping, got {type(cfg).__name__}")

    unknown = sorted(set(cfg) - set(valid_keys))
    if unknown:
        raise SystemExit(
            f"{path}: unknown key(s) {unknown}. Valid keys: {sorted(valid_keys)}"
        )
    return cfg


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> int:
    """Seed python/numpy/torch. Decoding is greedy, so this mainly pins data order."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def _git(*args, repo_root=None):
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info(repo_root=None) -> dict:
    """Commit, branch and dirty flag. A dirty tree makes a run non-reconstructable."""
    status = _git("status", "--porcelain", repo_root=repo_root)
    return {
        "commit": _git("rev-parse", "HEAD", repo_root=repo_root),
        "commit_short": _git("rev-parse", "--short", "HEAD", repo_root=repo_root),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo_root=repo_root),
        "dirty": bool(status) if status is not None else None,
    }


def env_info() -> dict:
    """Versions and GPU identity — the first things to diff when numbers mismatch."""
    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:
        tf_version = None

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "name": p.name,
                "total_memory_gb": round(p.total_memory / 1024 ** 3, 2),
                "capability": f"{p.major}.{p.minor}",
            })

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "transformers": tf_version,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpus": gpus,
    }


# ---------------------------------------------------------------------------
# Hook timing
# ---------------------------------------------------------------------------

class TimedHook:
    """
    Wrap a kv_hook so eviction cost is measured separately from generation.

    CUDA is asynchronous, so each call is bracketed with cuda.synchronize(); without
    that we would be timing kernel launches rather than execution.
    """

    def __init__(self, hook):
        self._hook = hook
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, state):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()
        out = self._hook(state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds += perf_counter() - t0
        self.calls += 1
        return out

    def stats(self) -> dict:
        return {
            "hook_calls": self.calls,
            "hook_sec": round(self.seconds, 3),
            "hook_sec_per_call": round(self.seconds / self.calls, 4) if self.calls else None,
        }


# ---------------------------------------------------------------------------
# Run recording
# ---------------------------------------------------------------------------

class RunRecorder:
    """
    Times a run, captures peak GPU memory, and writes the results/<runID>/ tree:

        metrics.json   headline score plus run parameters (small, always committed)
        manifest.json  config path, git commit, seed, GPU, wall time, dataset slice
        raw/generations.jsonl.gz   per-sample output, with sha256 and size recorded
    """

    def __init__(self, run_id: str, results_root: str = "results"):
        self.run_id = run_id
        self.dir = os.path.join(results_root, run_id)
        self.raw_dir = os.path.join(self.dir, "raw")
        self._t0 = None
        self.wall_sec = None

    def __enter__(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_sec = round(perf_counter() - self._t0, 2)
        return False

    @staticmethod
    def _peak_gpu_gb():
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)

    def write_raw(self, results: list) -> dict:
        """Write per-sample records as gzipped JSONL; return sha256, size and path."""
        path = os.path.join(self.raw_dir, "generations.jsonl.gz")
        h = hashlib.sha256()
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for r in results:
                line = json.dumps(r, ensure_ascii=False)
                f.write(line + "\n")
                h.update(line.encode("utf-8"))

        size = os.path.getsize(path)
        size_mb = round(size / 1024 ** 2, 2)
        if size_mb > _MAX_RAW_MB:
            print(
                f"\nWARNING: {path} is {size_mb}MB (>{_MAX_RAW_MB}MB). Per CLAUDE.md do not "
                f"commit it — keep it on the pod; sha256 and the regeneration command are "
                f"recorded in the manifest."
            )
        return {
            "path": path.replace(os.sep, "/"),
            "sha256": h.hexdigest(),
            "bytes": size,
            "megabytes": size_mb,
            "committable": size_mb <= _MAX_RAW_MB,
        }

    def write(self, *, metrics: dict, config_path, seed: int, dataset_slice: dict,
              results: list, hook_stats=None, extra=None) -> dict:
        """Write metrics.json and manifest.json. Returns the manifest."""
        raw = self.write_raw(results)

        timing = {"wall_sec": self.wall_sec, "peak_gpu_gb": self._peak_gpu_gb()}
        if hook_stats:
            timing.update(hook_stats)

        metrics_out = dict(metrics)
        metrics_out.update({"run_id": self.run_id, "timing": timing})
        with open(os.path.join(self.dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2, ensure_ascii=False)

        manifest = {
            "run_id": self.run_id,
            "config_path": config_path,
            "command": " ".join(sys.argv),
            "regeneration_command": f"python {' '.join(sys.argv)}",
            "seed": seed,
            "git": git_info(),
            "env": env_info(),
            "timing": timing,
            "dataset_slice": dataset_slice,
            "raw": raw,
            **(extra or {}),
        }
        with open(os.path.join(self.dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        if manifest["git"]["dirty"]:
            print("\nWARNING: working tree is dirty — this run is not reconstructable "
                  "from its recorded commit.")
        return manifest
