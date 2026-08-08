# Project

Reproduction and analysis of counter-causal KV-cache eviction.

# CLAUDE.md — Counter-Causal KV Cache: Reproduction & Follow-up Experiments

## What this repo is
Fork of `metacognitionai/counter_causal` Purpose: (1) reproduce key results, (2) run three follow-up experiments (Q1–Q3 below).

## Research questions (one branch each. Do after reproduction done)
- **Q1 / `q1-layer-sweep`** — Is the last layer the right scoring layer for the fast
  variant? Generalize the H^(L-1) hook to arbitrary layer ℓ, map to logits via
  final-norm + LM head (logit lens). Sweep ℓ ∈ {8, 16, 24, 32} on Llama-3.1-8B.
- **Q2 / `q2-faithfulness`** — How well does the flipped-mask surprise ranking track a
  true backward conditional? 
- **Q3 / `q3-drift`** — Does score quality decay as fossil K,V accumulate across refresh
  cycles?

## Branch & commit policy
- `main`: reproduction runs only, plus CLAUDE.md / EXPERIMENTS.md / report. No method
  changes on main, ever.
- One branch per question. Branch from main. Never merge experiment branches into main
  before the report is written; the report links to branches.
- Every experiment branch gets `plans/<branch>.md` BEFORE code changes: hypothesis,
  protocol (≤10 lines), expected runs, expected outputs. Write the plan first, then code.
- Commit message format: `[runID] short description` (runID from EXPERIMENTS.md).
- **Push after every completed run.** The RunPod pod is ephemeral and can be reclaimed
  without warning. An unpushed result does not exist.

## Environment
- RunPod, 1× H100, image: PyTorch 2.10 + CUDA 12.8, Python 3.12.
- Precision: bf16 (repo default). Greedy decoding everywhere. Single seed (record it).
- Setup: `pip install -r requirements.txt`; models pulled from HF (Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct)
- Sanity check: run `full` then `sliding` on MATH500 with Qwen2.5-7B-Instruct first to ensure env setup. Comapre results with reference number.

## Reference numbers (paper, Table 2 & Fig. 5 — reproduction targets)
MATH500 (J=512, h=256, max 2048 tok, full 500 problems):
- Qwen2.5-7B:  full .766 | sliding .692 | H2O .762 | counter .744 | counter-fast .736
- Llama-3.1-8B: full .488 | sliding .458 | H2O .464 | counter .482 | counter-fast .480

## Repo layout (ours, on top of upstream)
- `plans/` — one pre-registered plan per experiment branch
- `configs/` — one YAML per run; never hand-edit constants in source for a run
- `results/<runID>/` — metrics.json, score dumps (.csv), manifest.json
  (manifest = config path, git commit, seed, GPU, wall time, dataset slice)
- `results/<runID>/raw/` — compressed generations (.jsonl.gz) ONLY if <50MB; otherwise
  keep on pod, record sha256 + regeneration command in manifest
- `report/`

## Rules for Claude Code
1. Never modify upstream scoring/eviction logic on `main`.
2. Any code change: new branch + plan file first. No exceptions, including "tiny" changes.
3. After every run: write `results/<runID>/manifest.json`, append the EXPERIMENTS.md
   ledger row, commit, push. One run = one ledger row = one commit, even failed runs.
4. Do not commit: model weights, HF cache, anything >50MB, tokens/credentials.
5. When reproducing: change nothing but the config. When numbers mismatch, diff env
   (torch/transformers versions, dtype, subset) before touching code.
6. Terminology in all docs and code: "counter-causal (full)" and "counter-causal (fast)".