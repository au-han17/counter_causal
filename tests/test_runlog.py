"""Smoke tests for runlog.py that do not require loading a model."""
import json, os, sys, tempfile, traceback

sys.path.insert(0, os.path.abspath("."))
from runlog import load_config, seed_everything, TimedHook, RunRecorder, git_info, env_info

VALID = {"task", "model", "hook", "cache_size", "chunk_size", "max_new_tokens", "seed"}
ok = fail = 0


def check(name, fn):
    global ok, fail
    try:
        fn()
        print(f"  PASS  {name}")
        ok += 1
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        fail += 1


def t_config_loads():
    cfg = load_config("configs/r02.yaml", VALID)
    assert cfg["task"] == "math500", cfg
    assert cfg["cache_size"] == 512, cfg
    assert cfg["hook"] == "sliding", cfg


def t_unknown_key_rejected():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write("task: math500\ncach_size: 512\n")   # deliberate typo
        p = f.name
    try:
        load_config(p, VALID)
        raise AssertionError("typo'd key was accepted")
    except SystemExit as e:
        assert "cach_size" in str(e), e
    finally:
        os.unlink(p)


def t_seed():
    import torch
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    assert torch.equal(a, torch.randn(4))


def t_timed_hook():
    th = TimedHook(lambda s: s + 1)
    assert th(1) == 2 and th(2) == 3
    st = th.stats()
    assert st["hook_calls"] == 2 and st["hook_sec"] >= 0.0


def t_recorder():
    with tempfile.TemporaryDirectory() as d:
        with RunRecorder("R-TEST", results_root=d) as rec:
            pass
        m = rec.write(
            metrics={"score_key": "accuracy", "score": 0.5},
            config_path="configs/r02.yaml",
            seed=0,
            dataset_slice={"task": "math500", "n_samples": 2},
            results=[{"idx": 0, "pred": "a"}, {"idx": 1, "pred": "b"}],
            hook_stats=TimedHook(lambda s: s).stats(),
        )
        base = os.path.join(d, "R-TEST")
        for f in ("metrics.json", "manifest.json", os.path.join("raw", "generations.jsonl.gz")):
            assert os.path.exists(os.path.join(base, f)), f
        for field in ("run_id", "config_path", "seed", "git", "env", "timing",
                      "dataset_slice", "raw", "command", "regeneration_command"):
            assert field in m, field
        assert m["git"]["commit"], "git commit not captured"
        assert m["env"]["torch"], "torch version not captured"
        assert m["timing"]["wall_sec"] is not None
        assert len(m["raw"]["sha256"]) == 64
        metrics = json.load(open(os.path.join(base, "metrics.json"), encoding="utf-8"))
        assert "results" not in metrics, "per-sample data leaked into metrics.json"
        print("     manifest keys:", sorted(m))
        print("     env:", {k: m["env"][k] for k in ("torch", "transformers", "cuda")})
        print("     gpu:", m["env"]["gpus"])


for name, fn in [
    ("config loads from YAML", t_config_loads),
    ("unknown config key rejected", t_unknown_key_rejected),
    ("seed is reproducible", t_seed),
    ("TimedHook counts calls", t_timed_hook),
    ("RunRecorder writes full tree", t_recorder),
]:
    check(name, fn)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
