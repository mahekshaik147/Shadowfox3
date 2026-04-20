"""Dataset loading and preprocessing for SST-2 sentiment classification."""
from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class SSTDataset(Dataset):
    """Wraps a HuggingFace SST-2 split into a tokenized PyTorch dataset."""

    def __init__(self, hf_split, tokenizer, max_len: int = 128):
        self.data = hf_split
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        enc = self.tokenizer(
            row["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
            "text": row["sentence"],
        }


def load_sst2(
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_len: int = 128,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Load SST-2 train / validation splits and a matching tokenizer.

    Note: the canonical SST-2 test set has hidden labels (it's part of GLUE),
    so we use the validation split for held-out evaluation as is standard
    practice in the literature.
    """
    raw = load_dataset("glue", "sst2")
    tok = AutoTokenizer.from_pretrained(model_name)

    train_ds = SSTDataset(raw["train"], tok, max_len)
    val_ds = SSTDataset(raw["validation"], tok, max_len)

    def _collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["label"] for b in batch]),
            "texts": [b["text"] for b in batch],
        }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, val_loader, tok
