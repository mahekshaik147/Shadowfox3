"""Adversarial perturbations and robustness probes."""
from __future__ import annotations
import random
import string
from typing import List

NEGATION_WORDS = ["not", "never", "no", "hardly", "barely"]


def char_typo(text: str, p: float = 0.1, seed: int | None = None) -> str:
    """Random character swap / drop / insert with probability `p` per char."""
    rng = random.Random(seed)
    out: List[str] = []
    for ch in text:
        if ch.isalpha() and rng.random() < p:
            op = rng.choice(["swap", "drop", "insert"])
            if op == "swap":
                out.append(rng.choice(string.ascii_lowercase))
            elif op == "drop":
                continue
            else:
                out.append(ch)
                out.append(rng.choice(string.ascii_lowercase))
        else:
            out.append(ch)
    return "".join(out)


def add_negation(text: str) -> str:
    """Naive negation injection: prepend 'It is not true that ...'."""
    return "It is not true that " + text.rstrip(". ") + "."


def double_negation(text: str) -> str:
    return "I won't say it's not " + text.rstrip(". ") + "."


def length_pad(text: str, filler: str = "Anyway, ", n: int = 10) -> str:
    return filler * n + text
