"""Training loop and evaluation utilities."""
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, mask)
        logits = out["logits"]
        total_loss += loss_fn(logits, labels).item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": p,
        "recall": r,
        "f1": f1,
        "preds": all_preds,
        "labels": all_labels,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    device: torch.device | None = None,
    log_every: int = 50,
) -> Dict[str, List[float]]:
    """Standard fine-tuning loop with linear warmup + AdamW."""
    from transformers import get_linear_schedule_with_warmup

    device = device or get_device()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids, mask)
            loss = loss_fn(out["logits"], labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            if (step + 1) % log_every == 0:
                pbar.set_postfix(loss=running / (step + 1))

        val = evaluate(model, val_loader, device)
        history["train_loss"].append(running / max(1, len(train_loader)))
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1"])
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={history['train_loss'][-1]:.4f}  "
            f"val_loss={val['loss']:.4f}  "
            f"val_acc={val['accuracy']:.4f}  "
            f"val_f1={val['f1']:.4f}"
        )
    return history
