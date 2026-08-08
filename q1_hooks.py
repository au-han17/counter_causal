"""
Q1 layer-sweep hook: counter-causal surprise scoring with the mask flipped only from
layer ℓ upward.

The two upstream hooks are endpoints of one family. Writing ℓ for the first layer whose
attention is flipped during the scoring pass:

    counter_causal_hook       ℓ = 0    every layer counter-causal
    counter_causal_fast_hook  ℓ = L-1  last layer only (and further approximated:
                                       FFN skipped, h^{(L-2)} reused, fp16 logits)

counter_causal_split_hook interpolates: layers 0..ℓ-1 attend with the ordinary causal
mask, layers ℓ..L-1 with the counter-causal mask. As in the full hook, the cached K/V
are reused at every layer, only Q is freshly projected, the FFN is applied normally,
and scores are fp32 logits of the actual token. ℓ=0 reproduces counter_causal_hook
exactly.

Kept in a separate module so upstream hooks.py stays untouched on this branch.
"""

import torch
import torch.nn.functional as F

from hooks import CacheState, _apply_rope, _get_cache_size, _select_tokens


def counter_causal_split_hook(model, flip_from_layer: int, cache_size: int = 2048,
                              frozen_size: int = 0, recent_size: int = 0, min_gap: int = 0):
    """
    Counter-causal surprise eviction with a per-layer mask split.

    Args:
        model:           HuggingFace causal LM (must match the one used for generation).
        flip_from_layer: First layer index scored with the counter-causal mask.
                         Layers below it use the ordinary causal mask (self included).
                         0 == counter_causal_hook; L-1 flips only the last layer.
        cache_size:      Maximum number of non-frozen tokens to retain.
        frozen_size:     Number of leading tokens never evicted (e.g. system prompt).
        recent_size:     Number of trailing tokens always kept without scoring.
        min_gap:         Block attention to the nearest min_gap neighbours in the
                         flipped layers (reduces K/V leakage). Causal layers ignore it.
    """
    assert recent_size < cache_size
    n_layers = len(model.model.layers)
    assert 0 <= flip_from_layer < n_layers, \
        f"flip_from_layer must be in [0, {n_layers - 1}], got {flip_from_layer}"
    surprise_size = cache_size - recent_size

    def hook(state: CacheState) -> CacheState:
        if _get_cache_size(state) <= cache_size + frozen_size:
            return state

        seq_len = state.tok.shape[1]
        body_size = seq_len - frozen_size
        device = state.tok.device

        cfg = model.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=0
        )
        cc_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1 + min_gap
        )

        with torch.no_grad():
            hidden = model.model.embed_tokens(state.tok)

            for layer_idx, layer in enumerate(model.model.layers):
                if isinstance(state.kv, tuple):
                    k = state.kv[layer_idx][0]
                    v = state.kv[layer_idx][1]
                else:
                    k = state.kv.layers[layer_idx].keys
                    v = state.kv.layers[layer_idx].values

                attn = layer.self_attn
                head_dim = attn.head_dim

                h_norm = layer.input_layernorm(hidden)
                q = attn.q_proj(h_norm)
                q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

                cos, sin = model.model.rotary_emb(h_norm, position_ids=state.pos)
                q = _apply_rope(q, cos, sin)

                if num_kv_heads < num_heads:
                    g = num_heads // num_kv_heads
                    k = k.repeat_interleave(g, dim=1)
                    v = v.repeat_interleave(g, dim=1)

                mask = causal_mask if layer_idx < flip_from_layer else cc_mask
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask.unsqueeze(0).unsqueeze(0),
                    scale=head_dim ** -0.5,
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0)
                attn_out = attn_out.transpose(1, 2).reshape(1, seq_len, -1)
                attn_out = attn.o_proj(attn_out)

                hidden = hidden + attn_out
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

            logits = model.lm_head(model.model.norm(hidden)).float()

        scores = torch.gather(logits, 2, state.tok.unsqueeze(-1)).squeeze(-1)[0]
        scores[-(min_gap + 1):] = float('-inf')

        non_recent_size = body_size - recent_size
        body_scores = scores[frozen_size:frozen_size + non_recent_size]
        surprise_indices = body_scores.topk(min(surprise_size, non_recent_size), largest=False).indices
        recent_indices = torch.arange(body_size - recent_size, body_size, device=device)
        keep = torch.cat([surprise_indices, recent_indices]).sort().values
        return _select_tokens(state, frozen_size, keep)

    return hook
