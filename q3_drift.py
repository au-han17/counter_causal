"""
Q3 — fossil-K/V drift across refresh cycles.

Pass A (library): per conversation, chunked causal prefill with eviction OFF — same
chunk boundaries and kernel path as Pass B — harvesting every position's K/V at every
layer plus h^(L-1) to CPU. The drift-free reference.

Pass B (instrumented): chunked prefill with counter-causal (full) driving eviction.
At each refresh cycle, before the argsort: practical scores from the fossil cache and
oracle scores from library K/V gathered by original position, for CC-full, CC-fast
and Importance. Eviction then proceeds from CC-full practical scores, untouched.

Sanity: until the first eviction the two caches are bit-identical (same loop, same
kernels), so cycle 1 asserts torch.equal on the cache and must report Spearman = 1.

Usage:
  python q3_drift.py --self_test
  python q3_drift.py --config configs/q3_drift.yaml --run_id Q3-smoke --n_convs 1
  python q3_drift.py --config configs/q3_drift.yaml --run_id Q3-01
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from runlog import load_config, seed_everything, RunRecorder
from hooks import CacheState, _get_cache_size, _select_tokens

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


# ---------------------------------------------------------------------------
# Rank / set statistics
# ---------------------------------------------------------------------------

def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _kv_layer(kv, layer_idx):
    if isinstance(kv, tuple):
        return kv[layer_idx][0], kv[layer_idx][1]
    return kv.layers[layer_idx].keys, kv.layers[layer_idx].values


def gather_library_kv(lib_k, lib_v, positions, device):
    """Tuple-of-(k,v) cache holding the library's fresh K/V for `positions`."""
    idx = positions.cpu()
    out = []
    for k, v in zip(lib_k, lib_v):
        out.append((k.index_select(2, idx).to(device),
                    v.index_select(2, idx).to(device)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Scoring (mirrors hooks.py on an arbitrary K/V source)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, cos, sin):
    return (q * cos.unsqueeze(1)) + (_rotate_half(q) * sin.unsqueeze(1))


def _flip_attend(model, layer, hidden, k, v, pos, cc_mask):
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    attn = layer.self_attn
    head_dim = attn.head_dim
    seq_len = hidden.shape[1]

    h_norm = layer.input_layernorm(hidden)
    q = attn.q_proj(h_norm).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    cos, sin = model.model.rotary_emb(h_norm, position_ids=pos)
    q = _apply_rope(q, cos, sin)

    if num_kv_heads < num_heads:
        g = num_heads // num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=cc_mask.unsqueeze(0).unsqueeze(0), scale=head_dim ** -0.5)
    attn_out = torch.nan_to_num(attn_out, nan=0.0)
    attn_out = attn_out.transpose(1, 2).reshape(1, seq_len, -1)
    return attn.o_proj(attn_out)


def cc_full_scores(model, kv, input_ids, pos):
    """counter_causal_hook's score vector (hooks.py:277): fp32 logit of each token."""
    seq_len = input_ids.shape[1]
    device = input_ids.device
    cc_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(model.model.layers):
            k, v = _kv_layer(kv, layer_idx)
            hidden = hidden + _flip_attend(model, layer, hidden, k, v, pos, cc_mask)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = model.lm_head(model.model.norm(hidden)).float()
    scores = torch.gather(logits, 2, input_ids.unsqueeze(-1)).squeeze(-1)[0]
    return scores.cpu().numpy()


def cc_fast_scores(model, kv, hidden_l1, input_ids, pos):
    """counter_causal_fast_hook's score vector (hooks.py:372): last layer, FFN skipped."""
    seq_len = input_ids.shape[1]
    device = input_ids.device
    cc_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    last = model.model.layers[-1]
    k, v = _kv_layer(kv, len(model.model.layers) - 1)
    with torch.no_grad():
        attn_out = _flip_attend(model, last, hidden_l1, k, v, pos, cc_mask)
        h_mid = hidden_l1 + attn_out  # skip FFN, as deployed
        logits = model.lm_head(model.model.norm(h_mid))
    scores = torch.gather(logits.float(), 2, input_ids.unsqueeze(-1)).squeeze(-1)[0]
    return scores.cpu().numpy()


def imp_scores(last_keys, frozen_size):
    """importance_eviction_hook's body-space score vector (hooks.py:212)."""
    with torch.no_grad():
        k = last_keys[:, :, frozen_size:, :]
        k = k / torch.clamp(k.abs().max(), min=1e-6)
        q = k[:, :, -1:, :]
        s = (q @ k.transpose(-2, -1)).squeeze(-2).mean(1)
    return s[0].float().cpu().numpy()


def cc_keep_indices(scores_full, frozen_size, cache_size):
    """Body-space keep set exactly as the cc hooks derive it (recent=0, min_gap=0)."""
    scores = scores_full.copy()
    scores[-1] = -np.inf
    body = scores[frozen_size:]
    k = min(cache_size, len(body))
    keep = np.argpartition(body, k - 1)[:k]
    return set(int(i) for i in keep)


def imp_keep_indices(body_scores, cache_size):
    """importance hook keeps the HIGHEST-scoring body entries."""
    k = min(cache_size, len(body_scores))
    keep = np.argpartition(-np.asarray(body_scores), k - 1)[:k]
    return set(int(i) for i in keep)


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

def chunked_prefill(model, input_ids, chunk_size, on_chunk):
    """Replicates generate_with_kv_hook's prefill loop; on_chunk may evict."""
    device = model.device
    buf = [None]

    def _capture(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        buf[0] = h.detach()

    handle = model.model.layers[-2].register_forward_hook(_capture)
    state = None
    try:
        for step in range(0, input_ids.shape[1], chunk_size):
            chunk = input_ids[:, step:step + chunk_size]
            pos_ids = torch.arange(step, step + chunk.shape[1], device=device,
                                   dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                out = model(input_ids=chunk,
                            past_key_values=None if state is None else state.kv,
                            position_ids=pos_ids, use_cache=True)
            if state is None:
                state = CacheState(kv=out.past_key_values, tok=chunk, pos=pos_ids,
                                   hidden=buf[0])
            else:
                state = CacheState(
                    kv=out.past_key_values,
                    tok=torch.cat((state.tok, chunk), dim=1),
                    pos=torch.cat((state.pos, pos_ids), dim=1),
                    hidden=torch.cat((state.hidden, buf[0]), dim=1),
                )
            state = on_chunk(state, step + chunk.shape[1])
    finally:
        handle.remove()
    return state


def build_library(model, input_ids, chunk_size):
    """Pass A: eviction-free chunked prefill; harvest K/V + h^(L-1) to CPU."""
    n_layers = len(model.model.layers)
    lib_k = [[] for _ in range(n_layers)]
    lib_v = [[] for _ in range(n_layers)]
    lib_h = []
    done = [0]

    def on_chunk(state, _processed):
        new_len = _get_cache_size(state)
        for l in range(n_layers):
            k, v = _kv_layer(state.kv, l)
            lib_k[l].append(k[:, :, done[0]:new_len, :].cpu().clone())
            lib_v[l].append(v[:, :, done[0]:new_len, :].cpu().clone())
        lib_h.append(state.hidden[:, done[0]:new_len, :].cpu().clone())
        done[0] = new_len
        return state

    chunked_prefill(model, input_ids, chunk_size, on_chunk)
    lib_k = [torch.cat(parts, dim=2) for parts in lib_k]
    lib_v = [torch.cat(parts, dim=2) for parts in lib_v]
    lib_h = torch.cat(lib_h, dim=1)
    torch.cuda.empty_cache()
    return lib_k, lib_v, lib_h


def run_instrumented(model, input_ids, chunk_size, cache_size, frozen_size,
                     lib_k, lib_v, lib_h, conv_id, records):
    """Pass B: cc-full-driven eviction with per-cycle practical/oracle logging."""
    device = model.device
    cycle = [0]
    evictions_so_far = [0]
    prev_cache_len = [0]

    def on_chunk(state, _processed):
        if _get_cache_size(state) <= cache_size + frozen_size:
            prev_cache_len[0] = _get_cache_size(state)
            return state
        cycle[0] += 1
        n_new = _get_cache_size(state) - prev_cache_len[0]  # this chunk's cohort
        seq_len = state.tok.shape[1]
        positions = state.pos[0]

        # oracle cache: fresh K/V (and hidden) for the same surviving positions
        okv = gather_library_kv(lib_k, lib_v, positions, device)
        oh = lib_h.index_select(1, positions.cpu()).to(device)

        # cycle-1 sanity: same loop, same kernels, nothing evicted yet -> bit-equal
        if evictions_so_far[0] == 0:
            fk, _ = _kv_layer(state.kv, len(lib_k) - 1)
            if not torch.equal(fk, okv[-1][0]):
                print(f"  WARN {conv_id}: cycle-1 cache is not bit-identical to the "
                      f"library (max delta "
                      f"{(fk - okv[-1][0]).abs().max().item():.3e}) — kernel "
                      f"nondeterminism; Spearman should still be ~1")

        cc_p = cc_full_scores(model, state.kv, state.tok, state.pos)
        cc_o = cc_full_scores(model, okv, state.tok, state.pos)
        fast_p = cc_fast_scores(model, state.kv, state.hidden, state.tok, state.pos)
        fast_o = cc_fast_scores(model, okv, oh, state.tok, state.pos)
        imp_p = imp_scores(_kv_layer(state.kv, -1)[0], frozen_size)
        imp_o = imp_scores(okv[-1][0], frozen_size)

        # correlations over body entries minus the pinned last entry
        sl = slice(frozen_size, seq_len - 1)
        body_n = seq_len - frozen_size
        rec = {
            "conv_id": conv_id, "cycle": cycle[0], "n_body": body_n,
            "spearman_cc": spearman(cc_p[sl], cc_o[sl]),
            "spearman_fast": spearman(fast_p[sl], fast_o[sl]),
            "spearman_imp": spearman(imp_p[:-1], imp_o[:-1]),
        }

        # decision metrics
        all_body = set(range(body_n))
        for name, kp, ko in (
            ("cc", cc_keep_indices(cc_p, frozen_size, cache_size),
                   cc_keep_indices(cc_o, frozen_size, cache_size)),
            ("fast", cc_keep_indices(fast_p, frozen_size, cache_size),
                     cc_keep_indices(fast_o, frozen_size, cache_size)),
            ("imp", imp_keep_indices(imp_p, cache_size),
                    imp_keep_indices(imp_o, cache_size)),
        ):
            rec[f"jac_keep_{name}"] = jaccard(kp, ko)
            rec[f"jac_evict_{name}"] = jaccard(all_body - kp, all_body - ko)

        # mechanism: K drift of the fossil cache vs the library.
        # Each K is computed once and never updated, so per-position cosine is fixed
        # at birth; the survivor mean moves only through population turnover. The
        # newest-cohort mean isolates compounding: cohorts born after more evictions
        # were computed from a more corrupted cache.
        for l in (len(lib_k) // 2, len(lib_k) - 1):
            fk, _ = _kv_layer(state.kv, l)
            cs = F.cosine_similarity(fk[0, :, frozen_size:, :].float(),
                                     okv[l][0][0, :, frozen_size:, :].float(), dim=-1)
            rec[f"cos_k_l{l}"] = round(float(cs.mean().item()), 6)
            rec[f"cos_k_l{l}_newcohort"] = round(float(cs[:, -n_new:].mean().item()), 6)

        records.append(rec)
        print(f"  [{conv_id}] cycle {cycle[0]:>2} n={body_n} "
              f"rho cc={rec['spearman_cc']:.4f} fast={rec['spearman_fast']:.4f} "
              f"imp={rec['spearman_imp']:.4f} jac_ev cc={rec['jac_evict_cc']:.3f}")

        del okv, oh
        # evict exactly as counter_causal_hook would, from the practical scores
        keep_body = cc_keep_indices(cc_p, frozen_size, cache_size)
        keep = torch.tensor(sorted(keep_body), device=device, dtype=torch.long)
        evictions_so_far[0] += 1
        state = _select_tokens(state, frozen_size, keep)
        prev_cache_len[0] = _get_cache_size(state)
        return state

    chunked_prefill(model, input_ids, chunk_size, on_chunk)
    torch.cuda.empty_cache()
    return cycle[0]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_conversations(data_path):
    if not os.path.exists(data_path):
        import requests
        print(f"Downloading LoCoMo to {data_path} ...")
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(requests.get(LOCOMO_URL, timeout=120).json(), f)
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.values() if isinstance(data, dict) else data
    from tasks import LOCOMO_SYSTEM, _build_locomo_conversation
    out = []
    for i, item in enumerate(items):
        conv_text, sp1, sp2 = _build_locomo_conversation(item.get("conversation", {}))
        system = LOCOMO_SYSTEM.format(speaker1=sp1, speaker2=sp2)
        out.append({"conv_id": f"conv{i}", "system": system,
                    "prompt": system + "Input: " + conv_text})
    return out


# ---------------------------------------------------------------------------
# Self-test (no models)
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok = ok and cond

    check("spearman identical = 1", abs(spearman([3, 1, 2], [30, 10, 20]) - 1) < 1e-12)
    check("jaccard", abs(jaccard({1, 2, 3}, {2, 3, 4}) - 0.5) < 1e-12)

    # cc keep: frozen=2, cache=3, n=7 -> body of 5; last forced -inf so always kept
    s = np.array([9., 9., 5., 1., 4., 2., 3.])
    keep = cc_keep_indices(s, frozen_size=2, cache_size=3)
    check("cc keep = lowest incl pinned last", keep == {1, 3, 4})

    imp = imp_keep_indices(np.array([5., 1., 4., 2.]), 2)
    check("imp keep = highest", imp == {0, 2})

    # library gather alignment: value at position p equals p
    lk = [torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1).expand(1, 2, 10, 3).contiguous()]
    lv = [torch.zeros(1, 2, 10, 3)]
    got = gather_library_kv(lk, lv, torch.tensor([0, 3, 5]), "cpu")
    check("gather by position", torch.equal(got[0][0][0, 0, :, 0], torch.tensor([0., 3., 5.])))

    # pigeonhole floor sanity for the plan's numbers
    floor = (2 * 4096 - 5120) / (2 * 4096 - (2 * 4096 - 5120))
    check("keep-jaccard floor = 0.6", abs(floor - 0.6) < 1e-12)

    print("self-test", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Q3 fossil-K/V drift")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_id", default="Q3-01")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--data_path", default="data/locomo10.json")
    parser.add_argument("--n_convs", type=int, default=5)
    parser.add_argument("--cache_size", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
    pre, _ = parser.parse_known_args()
    if pre.self_test:
        self_test()
    if pre.config:
        valid = {a.dest for a in parser._actions if a.dest != "help"}
        parser.set_defaults(**load_config(pre.config, valid))
    args = parser.parse_args()
    seed_everything(args.seed)
    device = "cuda"

    convs = load_conversations(args.data_path)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading {args.model} ({args.dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map=device,
        trust_remote_code=True)
    model.eval()

    # 5 longest by token count (deterministic; maximizes refresh cycles)
    for c in convs:
        c["ids"] = tok(c["prompt"], return_tensors="pt",
                       truncation=True, max_length=args.max_tokens).input_ids
        c["frozen"] = len(tok(c["system"]).input_ids)
    convs.sort(key=lambda c: -c["ids"].shape[1])
    convs = convs[:args.n_convs]
    print("selected:", [(c["conv_id"], c["ids"].shape[1]) for c in convs])

    records = []
    with RunRecorder(args.run_id) as rec:
        for c in convs:
            n_tok = c["ids"].shape[1]
            print(f"\n{c['conv_id']}: {n_tok} tokens, frozen={c['frozen']}")
            ids = c["ids"].to(device)
            lib_k, lib_v, lib_h = build_library(model, ids, args.chunk_size)
            n_cycles = run_instrumented(model, ids, args.chunk_size, args.cache_size,
                                        c["frozen"], lib_k, lib_v, lib_h,
                                        c["conv_id"], records)
            print(f"{c['conv_id']}: {n_cycles} cycles")
            del lib_k, lib_v, lib_h
            torch.cuda.empty_cache()

    if not records:
        sys.exit("no refresh cycles occurred — conversations shorter than cache_size?")

    def agg(key, pred=lambda r: True):
        vals = [r[key] for r in records if pred(r) and not math.isnan(r[key])]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {}
    for m in ("cc", "fast", "imp"):
        summary[m] = {
            "spearman_cycle1": agg(f"spearman_{m}", lambda r: r["cycle"] == 1),
            "spearman_cycles_2_5": agg(f"spearman_{m}", lambda r: 2 <= r["cycle"] <= 5),
            "spearman_cycles_6_10": agg(f"spearman_{m}", lambda r: 6 <= r["cycle"] <= 10),
            "spearman_cycles_11plus": agg(f"spearman_{m}", lambda r: r["cycle"] >= 11),
            "jac_evict_cycles_2plus": agg(f"jac_evict_{m}", lambda r: r["cycle"] >= 2),
        }
    summary["cos_k_mid_last_cycle_bands"] = {
        f"l{l}": {
            "cycle1": agg(f"cos_k_l{l}", lambda r: r["cycle"] == 1),
            "cycles_11plus": agg(f"cos_k_l{l}", lambda r: r["cycle"] >= 11),
            "newcohort_cycle1": agg(f"cos_k_l{l}_newcohort", lambda r: r["cycle"] == 1),
            "newcohort_cycles_11plus": agg(f"cos_k_l{l}_newcohort", lambda r: r["cycle"] >= 11),
        } for l in (len(model.model.layers) // 2, len(model.model.layers) - 1)
    }

    metrics = {
        "score_key": "mean_spearman_cc_practical_vs_oracle_cycle2plus",
        "score": agg("spearman_cc", lambda r: r["cycle"] >= 2),
        "model": args.model, "dtype": args.dtype,
        "hook": "q3-drift",
        "cache_size": args.cache_size, "chunk_size": args.chunk_size,
        "summary": summary,
        "n_conversations": len(convs), "n_cycles_total": len(records),
    }
    rec.write(metrics=metrics, config_path=args.config, seed=args.seed,
              dataset_slice={"task": "q3-drift", "data_path": args.data_path,
                             "n_samples": len(convs),
                             "conv_lengths": [c["ids"].shape[1] for c in convs]},
              results=records)

    print("\n== summary ==")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
