"""Attention-weight extraction and visualization."""
from __future__ import annotations
from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def get_attention(model, tokenizer, text: str, device, max_len: int = 64):
    """Run a single example and return (tokens, attentions).

    attentions: list of L tensors, each (n_heads, seq, seq).
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    out = model(enc["input_ids"], enc["attention_mask"], output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    atts = [a.squeeze(0).cpu() for a in out["attentions"]]
    return tokens, atts


def plot_head(tokens: List[str], att: torch.Tensor, layer: int, head: int, ax=None):
    """Plot a single (layer, head) attention matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        att[head].numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        ax=ax,
        cbar=False,
    )
    ax.set_title(f"Layer {layer} · Head {head}")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    return ax


def cls_attention_summary(attentions) -> np.ndarray:
    """How much attention does [CLS] put on each token, averaged over heads,
    per layer? Returns (num_layers, seq_len)."""
    out = []
    for layer_att in attentions:  # (heads, seq, seq)
        cls_row = layer_att[:, 0, :].mean(0).numpy()  # (seq,)
        out.append(cls_row)
    return np.stack(out)
