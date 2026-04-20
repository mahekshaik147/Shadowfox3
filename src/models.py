"""Model definitions: fine-tunable BERT classifier + classical baselines."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel


class BertSentimentClassifier(nn.Module):
    """BERT encoder + linear classification head over the [CLS] token.

    Mirrors `BertForSequenceClassification` but exposes hidden states and
    attention weights cleanly for analysis.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, output_attentions: bool = False,
                output_hidden_states: bool = False):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls))
        return {
            "logits": logits,
            "attentions": out.attentions,
            "hidden_states": out.hidden_states,
            "cls_embedding": cls,
        }


class BiLSTMClassifier(nn.Module):
    """Lightweight BiLSTM baseline. Takes pre-tokenized integer ids."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden: int = 128,
                 num_labels: int = 2, pad_idx: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        out, _ = self.lstm(x)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = out.mean(1)
        return {"logits": self.fc(pooled)}
