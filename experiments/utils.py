# -*- coding: utf-8 -*-
"""
Shared utilities for NeuOS experiments.
Includes embedding surgery and hook helpers from Aletheia.
"""
import torch

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]


def apply_surgery(model, tok, strength=2.0):
    """Aletheia embedding surgery: amplify numeric token embeddings."""
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)


def replace_last_token(output, donor_vec):
    """Replace last-token hidden state in layer output (handles tuple/tensor, 2D/3D)."""
    if isinstance(output, tuple):
        h = output[0].clone()
        if h.dim() == 3:
            h[0, -1, :] = donor_vec.to(h.dtype)
        elif h.dim() == 2:
            h[-1, :] = donor_vec.to(h.dtype)
        return (h,) + output[1:]
    else:
        h = output.clone()
        if h.dim() == 3:
            h[0, -1, :] = donor_vec.to(h.dtype)
        elif h.dim() == 2:
            h[-1, :] = donor_vec.to(h.dtype)
        return h


def get_last_token(output):
    """Get last token hidden state from layer output."""
    tensor = output[0] if isinstance(output, tuple) else output
    if tensor.dim() == 3:
        return tensor[0, -1, :].detach().clone()
    elif tensor.dim() == 2:
        return tensor[-1, :].detach().clone()
    return tensor.detach().clone()


def load_model(model_id='Qwen/Qwen2.5-0.5B', device='cuda', dtype=torch.float32, surgery=True):
    """Load model with optional embedding surgery."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=dtype
    ).to(device)
    model.eval()
    if surgery:
        apply_surgery(model, tok)
    return model, tok
