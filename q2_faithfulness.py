"""
Q2 — faithfulness of the counter-causal surprise score.

Compares the flipped-mask surprise scores (full and fast variants, mirroring hooks.py
exactly but on fresh K/V) against an external backward-conditional referee
(ModernBERT masked-LM given only future context), at word level across tokenizers.

Pipeline per sequence (a Qwen-generated MATH500 solution from R-00-qwen):
  1. causal Qwen forward  -> all-layer K/V, h^(L-1), causal logits (forward ceiling)
  2. flipped passes on those K/V -> full/fast scores, raw-logit + log-prob variants
  3. ModernBERT cloze per word -> backward ([MASK]*k + future) and forward
     (past + [MASK]*k) conditionals
  4. word-level alignment via character spans; same filtered word set for all vectors
  5. per-sequence Spearman + frequency partial + direction control

Usage:
  python q2_faithfulness.py --self_test                 # pure-math checks, no models
  python q2_faithfulness.py --config configs/q2_faithfulness.yaml --run_id Q2-smoke --n_seqs 2
  python q2_faithfulness.py --config configs/q2_faithfulness.yaml --run_id Q2-01
"""

import argparse
import gzip
import io
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from runlog import load_config, seed_everything, RunRecorder

WORD_RE = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Rank statistics (no scipy dependency)
# ---------------------------------------------------------------------------

def rankdata(x):
    """Average ranks (ties share their mean rank), 1-based."""
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


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman(a, b):
    return _pearson(rankdata(a), rankdata(b))


def partial_spearman(a, b, z):
    """Spearman of a,b after partialling out z (partial Pearson on ranks)."""
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    r_ab, r_az, r_bz = _pearson(ra, rb), _pearson(ra, rz), _pearson(rb, rz)
    denom = math.sqrt((1 - r_az ** 2) * (1 - r_bz ** 2))
    if not denom or any(math.isnan(v) for v in (r_ab, r_az, r_bz)):
        return float("nan")
    return (r_ab - r_az * r_bz) / denom


# ---------------------------------------------------------------------------
# Word alignment
# ---------------------------------------------------------------------------

def words_of(text):
    """Character spans of whitespace-delimited words."""
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def map_words_to_tokens(word_spans, token_offsets):
    """
    For each word span, the (contiguous) token indices whose char spans overlap it.
    Offsets are (start, end) pairs; zero-width offsets never match.
    """
    out = []
    for a, b in word_spans:
        idxs = [i for i, (s, e) in enumerate(token_offsets) if e > a and s < b]
        out.append(idxs)
    return out


# ---------------------------------------------------------------------------
# Qwen-side scores (mirror hooks.py on fresh K/V)
# ---------------------------------------------------------------------------

def _layer_kv(kv, layer_idx):
    if isinstance(kv, tuple):
        return kv[layer_idx][0], kv[layer_idx][1]
    return kv.layers[layer_idx].keys, kv.layers[layer_idx].values


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


def causal_collect(model, input_ids):
    """One causal forward: (past_key_values, h^(L-1), fp32 logits)."""
    buf = [None]

    def _capture(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        buf[0] = h.detach()

    handle = model.model.layers[-2].register_forward_hook(_capture)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
    finally:
        handle.remove()
    return out.past_key_values, buf[0], out.logits.float()


def _flip_attend(model, layer, hidden, k, v, pos, cc_mask):
    """One layer's flipped attention exactly as hooks.py does it."""
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    attn = layer.self_attn
    head_dim = attn.head_dim
    seq_len = hidden.shape[1]

    h_norm = layer.input_layernorm(hidden)
    q = attn.q_proj(h_norm)
    q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    cos, sin = model.model.rotary_emb(h_norm, position_ids=pos)
    q = _apply_rope(q, cos, sin)

    if num_kv_heads < num_heads:
        g = num_heads // num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=cc_mask.unsqueeze(0).unsqueeze(0), scale=head_dim ** -0.5
    )
    attn_out = torch.nan_to_num(attn_out, nan=0.0)
    attn_out = attn_out.transpose(1, 2).reshape(1, seq_len, -1)
    return attn.o_proj(attn_out)


def _token_scores(logits, input_ids):
    """(raw_logit, log_prob) of the actual token at each position, both fp32 CPU."""
    logits = logits.float()
    idx = input_ids.unsqueeze(-1)
    raw = torch.gather(logits, 2, idx).squeeze(-1)[0]
    logprob = torch.gather(torch.log_softmax(logits, dim=-1), 2, idx).squeeze(-1)[0]
    return raw.cpu().numpy(), logprob.cpu().numpy()


def cc_full_scores(model, kv, input_ids, pos):
    """Mirror counter_causal_hook (hooks.py:277): every layer flipped, fp32 logits."""
    seq_len = input_ids.shape[1]
    device = input_ids.device
    cc_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(model.model.layers):
            k, v = _layer_kv(kv, layer_idx)
            hidden = hidden + _flip_attend(model, layer, hidden, k, v, pos, cc_mask)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = model.lm_head(model.model.norm(hidden))
    return _token_scores(logits, input_ids)


def cc_fast_scores(model, kv, hidden_l1, input_ids, pos):
    """Mirror counter_causal_fast_hook (hooks.py:372): last layer flipped, FFN skipped."""
    seq_len = input_ids.shape[1]
    device = input_ids.device
    cc_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    last = model.model.layers[-1]
    k, v = _layer_kv(kv, len(model.model.layers) - 1)
    with torch.no_grad():
        attn_out = _flip_attend(model, last, hidden_l1, k, v, pos, cc_mask)
        h_mid = hidden_l1 + attn_out  # skip FFN, as deployed
        logits = model.lm_head(model.model.norm(h_mid))
    return _token_scores(logits, input_ids)


def causal_forward_scores(logits, input_ids):
    """log P(x_i | past) from causal logits; position 0 gets nan (no past)."""
    logprobs = torch.log_softmax(logits, dim=-1)
    n = input_ids.shape[1]
    out = np.full(n, np.nan)
    idx = input_ids[0, 1:].unsqueeze(-1)
    vals = torch.gather(logprobs[0, :-1, :], 1, idx).squeeze(-1)
    out[1:] = vals.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Referee (ModernBERT cloze)
# ---------------------------------------------------------------------------

def referee_scores(ref_model, ref_tok, ref_ids, word_tokidx, batch_size, device):
    """
    Per-word backward/forward conditional log-probs.

    backward: [CLS] [MASK]*k  future_tokens [SEP]   (past dropped, not masked)
    forward:  [CLS] past_tokens [MASK]*k  [SEP]
    Returns dict word_index -> {bwd_sum, bwd_mean, fwd_sum, fwd_mean}; words with an
    empty token set get no entry.
    """
    cls_id, sep_id = ref_tok.cls_token_id, ref_tok.sep_token_id
    mask_id, pad_id = ref_tok.mask_token_id, ref_tok.pad_token_id

    jobs = []  # (word_idx, direction, ids, mask_positions, true_ids)
    for w, idxs in enumerate(word_tokidx):
        if not idxs:
            continue
        t0, t1 = idxs[0], idxs[-1]
        k = t1 - t0 + 1
        true_ids = ref_ids[t0:t1 + 1]
        bwd = [cls_id] + [mask_id] * k + ref_ids[t1 + 1:] + [sep_id]
        fwd = [cls_id] + ref_ids[:t0] + [mask_id] * k + [sep_id]
        jobs.append((w, "bwd", bwd[:8000], list(range(1, 1 + k)), true_ids))
        fwd_mask_start = 1 + t0
        jobs.append((w, "fwd", fwd[:8000], list(range(fwd_mask_start, fwd_mask_start + k)), true_ids))

    results = {}
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]))
    for start in range(0, len(order), batch_size):
        chunk = [jobs[j] for j in order[start:start + batch_size]]
        maxlen = max(len(c[2]) for c in chunk)
        ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        att = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, (_, _, seq, _, _) in enumerate(chunk):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            att[r, :len(seq)] = 1
        with torch.no_grad():
            out = ref_model(input_ids=ids.to(device), attention_mask=att.to(device))
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        for r, (w, direction, _, mask_pos, true_ids) in enumerate(chunk):
            vals = [logp[r, p, t].item() for p, t in zip(mask_pos, true_ids)]
            rec = results.setdefault(w, {})
            rec[f"{direction}_sum"] = float(sum(vals))
            rec[f"{direction}_mean"] = float(sum(vals) / len(vals))
    return results


# ---------------------------------------------------------------------------
# Per-sequence pipeline
# ---------------------------------------------------------------------------

def analyze_sequence(text, seq_id, qwen, qwen_tok, ref_model, ref_tok, freq,
                     args, device, word_rows):
    enc = qwen_tok(text, return_offsets_mapping=True, add_special_tokens=False,
                   truncation=True, max_length=args.max_tokens)
    q_ids = enc["input_ids"]
    n = len(q_ids)
    if n < args.min_tokens:
        return None
    text = text[:enc["offset_mapping"][-1][1]]  # clip to what was tokenized

    renc = ref_tok(text, return_offsets_mapping=True, add_special_tokens=False,
                   truncation=True, max_length=8000)
    r_ids = renc["input_ids"]

    spans = words_of(text)
    q_map = map_words_to_tokens(spans, enc["offset_mapping"])
    r_map = map_words_to_tokens(spans, renc["offset_mapping"])

    # Uniform word filter: both sides tokenized, not touching Qwen token 0 or the last
    # token (no past / no future), same set for every vector.
    keep = [w for w in range(len(spans))
            if q_map[w] and r_map[w] and 0 not in q_map[w] and (n - 1) not in q_map[w]]
    if len(keep) < args.min_words:
        return None
    if len(keep) > args.max_words_per_seq:
        step = len(keep) / args.max_words_per_seq
        keep = [keep[int(i * step)] for i in range(args.max_words_per_seq)]

    # ---- Qwen side ----
    ids_t = torch.tensor([q_ids], dtype=torch.long, device=device)
    pos = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0)
    kv, h_l1, causal_logits = causal_collect(qwen, ids_t)
    full_logit, full_logprob = cc_full_scores(qwen, kv, ids_t, pos)
    fast_logit, fast_logprob = cc_fast_scores(qwen, kv, h_l1, ids_t, pos)
    qfwd = causal_forward_scores(causal_logits, ids_t)
    del kv, h_l1, causal_logits
    torch.cuda.empty_cache()

    # ---- referee ----
    ref = referee_scores(ref_model, ref_tok, r_ids, [r_map[w] for w in keep],
                         args.batch_size, device)
    # referee_scores keys words by position in the list it was given; pair explicitly
    # so a dropped word can never shift the alignment
    pairs = [(w, ref[j]) for j, w in enumerate(keep) if j in ref]

    # ---- aggregate to word level ----
    def agg(vec, idxs, how):
        vals = [vec[i] for i in idxs if not np.isnan(vec[i])]
        if not vals:
            return np.nan
        return float(np.mean(vals)) if how == "mean" else float(np.max(vals))

    rows = []
    for w, rec in pairs:
        qi = q_map[w]
        word = text[spans[w][0]:spans[w][1]]
        rows.append({
            "word": word,
            "full_logit": agg(full_logit, qi, "mean"),
            "full_logit_max": agg(full_logit, qi, "max"),
            "full_logprob": agg(full_logprob, qi, "mean"),
            "fast_logit": agg(fast_logit, qi, "mean"),
            "fast_logprob": agg(fast_logprob, qi, "mean"),
            "qwen_fwd": agg(qfwd, qi, "mean"),
            "ref_bwd": rec["bwd_mean"], "ref_bwd_sum": rec["bwd_sum"],
            "ref_fwd": rec["fwd_mean"],
            "neglogfreq": freq(word),
        })
    rows = [r for r in rows if not any(np.isnan(v) for k, v in r.items() if k != "word")]
    if len(rows) < args.min_words:
        return None

    col = lambda k: [r[k] for r in rows]
    out = {
        "seq_id": seq_id,
        "n_tokens": n,
        "n_words": len(rows),
        # primary + variants
        "rho_full_bwd": spearman(col("full_logit"), col("ref_bwd")),
        "rho_fast_bwd": spearman(col("fast_logit"), col("ref_bwd")),
        "rho_full_logprob_bwd": spearman(col("full_logprob"), col("ref_bwd")),
        "rho_fast_logprob_bwd": spearman(col("fast_logprob"), col("ref_bwd")),
        # controls
        "rho_ceiling_fwd": spearman(col("qwen_fwd"), col("ref_fwd")),
        "rho_full_reffwd": spearman(col("full_logit"), col("ref_fwd")),
        "rho_fast_reffwd": spearman(col("fast_logit"), col("ref_fwd")),
        "rho_full_freq": spearman(col("full_logit"), col("neglogfreq")),
        "rho_bwd_freq": spearman(col("ref_bwd"), col("neglogfreq")),
        "partial_full_bwd_freq": partial_spearman(col("full_logit"), col("ref_bwd"), col("neglogfreq")),
        "partial_fast_bwd_freq": partial_spearman(col("fast_logit"), col("ref_bwd"), col("neglogfreq")),
        # method-vs-method and sensitivity
        "rho_full_fast": spearman(col("full_logit"), col("fast_logit")),
        "rho_full_bwd_sumref": spearman(col("full_logit"), col("ref_bwd_sum")),
        "rho_fullmax_bwd": spearman(col("full_logit_max"), col("ref_bwd")),
    }
    for r in rows:
        r["seq_id"] = seq_id
    word_rows.extend(rows)
    return out


# ---------------------------------------------------------------------------
# Self-test (no models, no GPU)
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok = ok and cond

    check("rank ties", list(rankdata([10.0, 20.0, 20.0, 30.0])) == [1.0, 2.5, 2.5, 4.0])
    check("spearman monotone = 1", abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12)
    check("spearman reversed = -1", abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-12)
    check("spearman tie case = 0.866", abs(spearman([1, 2, 3], [2, 2, 5]) - 0.8660254) < 1e-6)

    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    a = z + 0.1 * rng.normal(size=500)
    b = z + 0.1 * rng.normal(size=500)
    check("partial kills confound", partial_spearman(a, b, z) < 0.5 < spearman(a, b))

    text = "ab cde f"
    spans = words_of(text)
    check("word spans", spans == [(0, 2), (3, 6), (7, 8)])
    offs = [(0, 1), (1, 2), (3, 5), (5, 6), (7, 8)]
    check("token map", map_words_to_tokens(spans, offs) == [[0, 1], [2, 3], [4]])
    check("zero-width offset ignored",
          map_words_to_tokens([(0, 2)], [(0, 0), (0, 2)]) == [[1]])

    print("self-test", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Q2 surprise-score faithfulness")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run_id", default="Q2-01")
    parser.add_argument("--dump", default="results/R-00-qwen/raw/generations.jsonl.gz")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--referee_model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16",
                        help="bfloat16 matches the R-00-qwen generation run")
    parser.add_argument("--n_seqs", type=int, default=50)
    parser.add_argument("--min_tokens", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_words_per_seq", type=int, default=400)
    parser.add_argument("--min_words", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
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

    print(f"Loading sequences from {args.dump}")
    texts = []
    with gzip.open(args.dump, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append((r.get("id", len(texts)), r["pred_raw"]))

    # word-frequency table over the whole dump (control variable)
    counts = Counter(w.lower() for _, t in texts for w in WORD_RE.findall(t))
    total = sum(counts.values())
    vocab = len(counts)

    def freq(word):
        return -math.log((counts.get(word.lower(), 0) + 1) / (total + vocab))

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
    print(f"Loading {args.qwen_model} ({args.dtype}) ...")
    qwen_tok = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        args.qwen_model, dtype=getattr(torch, args.dtype), device_map=device,
        trust_remote_code=True)
    qwen.eval()
    print(f"Loading {args.referee_model} ...")
    ref_tok = AutoTokenizer.from_pretrained(args.referee_model)
    ref_model = AutoModelForMaskedLM.from_pretrained(
        args.referee_model, dtype=torch.bfloat16, device_map=device)
    ref_model.eval()

    per_seq, word_rows = [], []
    with RunRecorder(args.run_id) as rec:
        for seq_id, text in texts:
            if len(per_seq) >= args.n_seqs:
                break
            try:
                r = analyze_sequence(text, seq_id, qwen, qwen_tok, ref_model, ref_tok,
                                     freq, args, device, word_rows)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM on {seq_id}, skipped")
                continue
            if r is None:
                continue
            per_seq.append(r)
            print(f"[{len(per_seq)}/{args.n_seqs}] {seq_id}: n_words={r['n_words']} "
                  f"full-bwd={r['rho_full_bwd']:.3f} fast-bwd={r['rho_fast_bwd']:.3f} "
                  f"ceiling={r['rho_ceiling_fwd']:.3f}")

    if not per_seq:
        sys.exit("no sequence produced enough scored words")

    keys = [k for k in per_seq[0] if k.startswith(("rho_", "partial_"))]
    summary = {}
    for k in keys:
        vals = np.array([s[k] for s in per_seq if not math.isnan(s[k])])
        summary[k] = {"mean": round(float(vals.mean()), 4),
                      "median": round(float(np.median(vals)), 4),
                      "std": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None,
                      "frac_positive": round(float((vals > 0).mean()), 3),
                      "n": int(len(vals))}

    headline = summary["rho_full_bwd"]["mean"]
    metrics = {
        "score_key": "mean_spearman_full_logit_vs_referee_bwd",
        "score": headline,
        "model": args.qwen_model,
        "referee": args.referee_model,
        "dtype": args.dtype,
        "hook": "q2-faithfulness",
        "summary": summary,
        "n_seqs": len(per_seq),
        "n_words_total": len(word_rows),
    }

    # word-level table for plotting
    os.makedirs(os.path.join(rec.dir, "raw"), exist_ok=True)
    wpath = os.path.join(rec.dir, "raw", "words.csv.gz")
    cols = list(word_rows[0].keys())
    with gzip.open(wpath, "wt", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in word_rows:
            f.write(",".join(json.dumps(r[c], ensure_ascii=False) if c == "word"
                             else f"{r[c]}" for c in cols) + "\n")

    rec.write(metrics=metrics, config_path=args.config, seed=args.seed,
              dataset_slice={"task": "q2-faithfulness", "source_dump": args.dump,
                             "n_samples": len(per_seq),
                             "min_tokens": args.min_tokens, "max_tokens": args.max_tokens,
                             "max_words_per_seq": args.max_words_per_seq},
              results=per_seq)

    print("\n== summary (mean per-sequence Spearman) ==")
    for k in keys:
        s = summary[k]
        print(f"  {k:26} {s['mean']:+.4f}  (median {s['median']:+.4f}, "
              f"{s['frac_positive']:.0%} positive, n={s['n']})")


if __name__ == "__main__":
    main()
