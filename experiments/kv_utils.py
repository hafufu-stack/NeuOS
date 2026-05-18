"""KV cache utilities for DynamicCache (Qwen2.5 / transformers 5.x)

IMPORTANT: When using past_key_values for continuation, you must:
1. Pass only the NEW tokens as input_ids (not the full sequence)
2. Let the model handle position_ids automatically (transformers 5.x does this)
3. The model will infer the correct sequence position from the KV cache length
"""
import torch
from transformers.cache_utils import DynamicCache


def swap_out(kv_cache):
    """Save DynamicCache to CPU RAM."""
    saved = []
    for layer_data in kv_cache:
        # layer_data is a tuple of (key, value, maybe_None)
        saved.append((layer_data[0].cpu().clone(), layer_data[1].cpu().clone()))
    return saved


def swap_in(saved_list, device='cuda'):
    """Restore DynamicCache from saved CPU tensors."""
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(saved_list):
        cache.update(k.to(device), v.to(device), layer_idx)
    return cache


def run_prefix(model, tok, prefix_text, device='cuda'):
    """Run prefix, return (logits, kv_cache)."""
    inp = tok(prefix_text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(**inp, use_cache=True)
    return out.logits, out.past_key_values


def continue_with_cache(model, tok, cont_text, kv_cache, device='cuda'):
    """Continue generation with existing KV cache. Returns (logits, new_kv)."""
    inp = tok(cont_text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, past_key_values=kv_cache, use_cache=True)
    return out.logits, out.past_key_values


def continue_from_saved(model, tok, cont_text, saved_kv, device='cuda'):
    """Swap in saved KV cache and continue. Returns (logits, new_kv)."""
    kv_gpu = swap_in(saved_kv, device)
    return continue_with_cache(model, tok, cont_text, kv_gpu, device)
