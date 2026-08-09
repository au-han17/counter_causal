"""Validate every configs/*.yaml against evaluate.py's actual argparse definition."""
import glob, io, os, sys, yaml

sys.path.insert(0, os.path.abspath("."))
sys.argv = ["evaluate.py"]

import argparse, evaluate  # noqa

# Rebuild the parser exactly as evaluate.main does, without running it.
src = io.open("evaluate.py", encoding="utf-8").read()
start = src.index('parser = argparse.ArgumentParser(description="KV cache eviction')
end = src.index('if pre_args.config:')
import textwrap
block = textwrap.dedent("    " + src[start:end])   # re-indent line 1, then dedent
ns = {"argparse": argparse}
exec(block, ns)                               # noqa: S102 - deliberate
parser = ns["parser"]

valid = {a.dest for a in parser._actions if a.dest != "help"}
choices = {a.dest: a.choices for a in parser._actions if a.choices}
types = {a.dest: a.type for a in parser._actions if a.type}

bad = 0
for path in sorted(glob.glob("configs/*.yaml")):
    if os.path.basename(path).startswith(("q2_", "q3_")):
        # standalone-pipeline configs; validated by their own script's load_config
        print(f"  skip  {os.path.basename(path):22} (standalone pipeline)")
        continue
    cfg = yaml.safe_load(io.open(path, encoding="utf-8")) or {}
    errs = []
    for k, v in cfg.items():
        if k not in valid:
            errs.append(f"unknown key {k!r}")
        elif k in choices and v not in choices[k]:
            errs.append(f"{k}={v!r} not in {list(choices[k])}")
        elif k in types and types[k] is int and not isinstance(v, int):
            errs.append(f"{k}={v!r} should be int")
    if errs:
        bad += 1
        print(f"  FAIL  {path}")
        for e in errs:
            print(f"          {e}")
    else:
        summary = " ".join(f"{k}={cfg[k]}" for k in ("task", "hook", "cache_size", "dtype")
                           if k in cfg)
        print(f"  ok    {os.path.basename(path):22} {summary}")

print(f"\n{len(glob.glob('configs/*.yaml')) - bad} valid, {bad} invalid")
sys.exit(1 if bad else 0)
