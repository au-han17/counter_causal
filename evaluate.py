"""
Evaluation script for "Back from the Future: Key-Value Cache Management by Counter-Causal Surprise."
Copyright (c) 2026, Metacognition (http://metacognitionai.com)

Example usage:
  python evaluate.py --task math500 --hook none --out results_none.json
  python evaluate.py --task math500 --hook counter_causal --cache_size 512 --chunk_size 256 --out results.json
  python evaluate.py --task math500 --hook counter_fast --cache_size 512 --chunk_size 256 --out results.json
  python evaluate.py --task longhealth --hook counter_causal --cache_size 4096 --auto_frozen --out results.json
  python evaluate.py --task math500 --hook none --dtype bfloat16 --out results_bf16.json
  python evaluate.py --config configs/r00_qwen.yaml --run_id R-00-qwen
"""

import json
import argparse
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from runlog import load_config, seed_everything, TimedHook, RunRecorder

from hooks import (
    sliding_window_hook,
    importance_eviction_hook,
    h2o_eviction_hook,
    counter_causal_hook,
    counter_causal_fast_hook,
)
from tasks import (
    MATH500_SYSTEM,
    LONGHEALTH_SYSTEM,
    QASPER_SYSTEM,
    LOCOMO_SYSTEM,
    run_math500_task,
    run_longhealth_task,
    run_qasper_task,
    run_locomo_task,
)


def build_hook(args, model):
    rs = args.recent_size
    fs = args.frozen_size
    cs = args.cache_size

    if args.hook == "sliding":
        return sliding_window_hook(window_size=cs, frozen_size=fs)
    if args.hook == "importance":
        return importance_eviction_hook(cache_size=cs, frozen_size=fs)
    if args.hook == "h2o":
        recent = rs if rs is not None else cs // 2
        return h2o_eviction_hook(cache_size=cs, recent_size=recent, frozen_size=fs)
    if args.hook == "counter_causal":
        recent = rs if rs is not None else 0
        return counter_causal_hook(model, cache_size=cs, recent_size=recent, frozen_size=fs)
    if args.hook == "counter_fast":
        recent = rs if rs is not None else 0
        return counter_causal_fast_hook(model, cache_size=cs, recent_size=recent, frozen_size=fs)
    return None  # "none" — full cache, no eviction


def main():
    # Pre-parse --config only, so its contents can seed the real parser's defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    parser = argparse.ArgumentParser(description="KV cache eviction evaluation")
    parser.add_argument("--config", default=None,
                        help="Run YAML supplying defaults; explicit CLI flags override it")
    parser.add_argument("--run_id", default=None,
                        help="Ledger run ID (e.g. R-01). Writes results/<run_id>/ with "
                             "metrics.json, manifest.json and raw/generations.jsonl.gz")
    parser.add_argument("--seed", type=int, default=0, help="Seed; recorded in the manifest")
    # --task is validated after parsing so a config file can supply it
    parser.add_argument("--task", choices=["math500", "longhealth", "qasper", "locomo"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                        help="Weight dtype. Default float16 matches upstream; both target "
                             "checkpoints are natively bfloat16")
    parser.add_argument("--hook", choices=["none", "sliding", "importance", "h2o", "counter_causal", "counter_fast"], default="none")
    parser.add_argument("--refresh_mode", choices=["chunked", "prefill_end"], default="chunked")
    parser.add_argument("--frozen_size", type=int, default=0, help="Leading tokens never evicted (e.g. system prompt length)")
    parser.add_argument("--auto_frozen", action="store_true", help="Set frozen_size automatically from the task system prompt")
    parser.add_argument("--cache_size", type=int, default=None, help="Maximum KV cache size J (default: 512 for math500, 4096 for others)")
    parser.add_argument("--recent_size", type=int, default=None, help="Trailing tokens always kept (default: cache_size//2 for h2o, 0 otherwise)")
    parser.add_argument("--chunk_size", type=int, default=None,  help="Tokens processed between hook calls (default: cache_size//4)")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    # task-specific filters
    parser.add_argument("--subject", default=None, help="MATH500: filter by subject")
    parser.add_argument("--level", type=int, default=None, help="MATH500: filter by difficulty level (1–5)")
    parser.add_argument("--category", type=int, default=None, help="LoCoMo: filter by question category (1–5)")
    parser.add_argument("--max_samples", type=int, default=None, help="LoCoMo: cap number of samples")
    parser.add_argument("--data_path", default=None, help="LoCoMo: local path to locomo10.json")
    # debugging
    parser.add_argument("--single_sample", type=int, default=None, help="Run on a single sample index (for debugging)")
    parser.add_argument("--out", default="results.json")

    if pre_args.config:
        valid_keys = {a.dest for a in parser._actions if a.dest != "help"}
        parser.set_defaults(**load_config(pre_args.config, valid_keys))

    args = parser.parse_args()
    if args.task is None:
        parser.error("--task is required (pass it directly or set it in --config)")

    seed_everything(args.seed)

    # ---- defaults ----
    if args.cache_size is None:
        args.cache_size = 512 if args.task == "math500" else 4096
    if args.chunk_size is None:
        args.chunk_size = args.cache_size // 4

    # ---- print settings ----
    # Printed before the model loads so a bad config fails fast rather than after a
    # multi-GB download. --auto_frozen adjusts cache_size later and logs the change.
    print(f"\nSettings:")
    print(f"  task:         {args.task}")
    print(f"  model:        {args.model}")
    print(f"  dtype:        {args.dtype}")
    print(f"  hook:         {args.hook}")
    print(f"  cache_size:   {args.cache_size}")
    print(f"  chunk_size:   {args.chunk_size}")
    print(f"  frozen_size:  {args.frozen_size}")
    print(f"  recent_size:  {args.recent_size}")
    print(f"  refresh_mode: {args.refresh_mode}")
    print(f"  seed:         {args.seed}")
    print(f"  config:       {args.config or '(none)'}")
    print(f"  run_id:       {args.run_id or '(none, writing to --out)'}")
    print()

    # ---- model ----
    print(f"Loading {args.model} ({args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda", trust_remote_code=True
    )
    model.eval()
    # Confirm what actually landed on the GPU, not just what was requested.
    print(f"Loaded. resolved weight dtype: {model.dtype}")

    # ---- frozen size from system prompt ----
    if args.auto_frozen:
        system_prompts = {
            "math500":    MATH500_SYSTEM,
            "longhealth": LONGHEALTH_SYSTEM,
            "qasper":     QASPER_SYSTEM,
            "locomo":     LOCOMO_SYSTEM.format(speaker1="", speaker2=""),
        }
        args.frozen_size = len(tokenizer.encode(system_prompts[args.task]))
        args.cache_size -= args.frozen_size
        assert args.cache_size > 0, "cache_size too small after subtracting frozen_size"
        print(f"auto_frozen: frozen_size={args.frozen_size}, adjusted cache_size={args.cache_size}")

    # ---- hook ----
    hook = build_hook(args, model)
    timed_hook = None
    if hook is not None:
        timed_hook = TimedHook(hook)
        hook = timed_hook

    # ---- run task ----
    recorder = RunRecorder(args.run_id) if args.run_id else None

    common = dict(
        model=model,
        tokenizer=tokenizer,
        kv_hook=hook,
        chunk_size=args.chunk_size,
        sample_index=args.single_sample,
        refresh_mode=args.refresh_mode,
    )

    with recorder if recorder is not None else nullcontext():
        if args.task == "math500":
            results, score = run_math500_task(
                **common,
                max_new_tokens=args.max_new_tokens or 2048,
                subject=args.subject,
                level=args.level,
            )
            output = {"score_key": "accuracy", "score": score, "results": results}

        elif args.task == "longhealth":
            results, score = run_longhealth_task(
                **common,
                max_new_tokens=args.max_new_tokens or 16,
            )
            output = {"score_key": "accuracy", "score": score, "results": results}

        elif args.task == "qasper":
            results, score = run_qasper_task(
                **common,
                max_new_tokens=args.max_new_tokens or 128,
            )
            output = {"score_key": "mean_f1", "score": score, "results": results}

        elif args.task == "locomo":
            results, score, per_cat = run_locomo_task(
                **common,
                max_new_tokens=args.max_new_tokens or 32,
                category=args.category,
                data_path=args.data_path,
                max_samples=args.max_samples,
            )
            output = {"score_key": "mean_f1", "score": score, "per_category": per_cat, "results": results}

    output.update({
        "task": args.task,
        "model": args.model,
        "dtype": str(model.dtype),
        "hook": args.hook,
        "refresh_mode": args.refresh_mode,
        "cache_size": args.cache_size,
        "chunk_size": args.chunk_size,
        "frozen_size": args.frozen_size,
        "recent_size": args.recent_size,
        "seed": args.seed,
        "run_id": args.run_id,
        "config": args.config,
    })

    if recorder is not None:
        metrics = {k: v for k, v in output.items() if k != "results"}
        manifest = recorder.write(
            metrics=metrics,
            config_path=args.config,
            seed=args.seed,
            dataset_slice={
                "task": args.task,
                "n_samples": len(results),
                "subject": args.subject,
                "level": args.level,
                "category": args.category,
                "max_samples": args.max_samples,
                "single_sample": args.single_sample,
            },
            results=results,
            hook_stats=timed_hook.stats() if timed_hook is not None else None,
        )
        t = manifest["timing"]
        print(f"\nSaved {len(results)} results to {recorder.dir}/")
        print(f"  wall {t['wall_sec']}s | peak GPU {t['peak_gpu_gb']}GB"
              + (f" | hook {t['hook_sec']}s over {t['hook_calls']} calls" if timed_hook else ""))
    else:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
