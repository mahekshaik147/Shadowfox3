"""Layer-wise linear probing of BERT representations."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm


@torch.no_grad()
def extract_layer_cls(model, loader, device, max_batches: int | None = None):
    """Extract [CLS] vectors at every layer for a dataset.

    Returns:
        feats: dict {layer_idx -> np.ndarray (N, hidden)}
        labels: np.ndarray (N,)
    """
    model.eval()
    by_layer: Dict[int, List[np.ndarray]] = {}
    labels: List[int] = []
    for i, batch in enumerate(tqdm(loader, desc="extracting")):
        if max_batches and i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model(ids, mask, output_hidden_states=True)
        # tuple of (L+1) tensors (B, T, H), first is embeddings
        for layer_idx, h in enumerate(out["hidden_states"]):
            by_layer.setdefault(layer_idx, []).append(h[:, 0, :].cpu().numpy())
        labels.extend(batch["labels"].cpu().tolist())
    feats = {k: np.concatenate(v, axis=0) for k, v in by_layer.items()}
    return feats, np.array(labels)


def probe_layers(train_feats, train_y, val_feats, val_y, C: float = 1.0):
    """Fit a logistic-regression probe at each layer; return accuracies."""
    accs = {}
    for layer in sorted(train_feats.keys()):
        clf = LogisticRegression(max_iter=1000, C=C, n_jobs=-1)
        clf.fit(train_feats[layer], train_y)
        preds = clf.predict(val_feats[layer])
        accs[layer] = accuracy_score(val_y, preds)
    return accs
