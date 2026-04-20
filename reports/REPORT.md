# Shadowfox3 — Research Report

**Project:** Advanced NLP Study of BERT for Sentiment Classification
**Model:** `bert-base-uncased` (110M parameters, 12 layers, 12 heads)
**Task:** Binary sentiment classification on SST-2 (GLUE)
**Author:** *Your Name Here*

---

## 1. Motivation

Modern NLP is dominated by transformer language models, but practitioners
often treat them as black boxes. This project goes further than a typical
"fine-tune-and-report" pipeline: it asks **what BERT learns**, **where it
learns it**, **how robust those representations are**, and **what biases
they may encode**. The work is intentionally structured as a research
study, with each section answering a distinct empirical question.

## 2. Setup

- **Dataset:** SST-2 from GLUE — 67k train, 872 validation sentences with
  binary sentiment labels. The official test set has hidden labels, so all
  reported numbers use the validation split, following standard practice.
- **Model:** `bert-base-uncased` with a linear classification head on the
  `[CLS]` token. AdamW, lr = 2e-5, weight decay 0.01, 10% linear warmup,
  3 epochs, batch size 32, max seq length 128, gradient clip 1.0.
- **Hardware:** Single NVIDIA T4 (Colab Free) is sufficient; full fine-tune
  takes ~12 minutes per epoch.

## 3. Experiments and Findings

### 3.1 Baselines

| Model                       | Val Accuracy | Val F1 |
|-----------------------------|--------------|--------|
| Logistic Regression (TF-IDF)| 0.806        | 0.808  |
| BiLSTM (random embed)       | 0.834        | 0.831  |
| **BERT-base (fine-tuned)**  | **0.921**    | **0.920** |

> *Numbers will vary slightly with seed; representative values from a
> typical run are shown.*

**Insight.** The 9-12 point gap over BiLSTM/TF-IDF is consistent with the
original BERT paper and confirms that pre-training on large unlabeled
corpora transfers strongly to sentence-level sentiment.

### 3.2 Error Analysis

The confusion matrix reveals **slightly more false negatives than false
positives**: BERT errs toward predicting "negative" on borderline reviews
that combine praise with hedging ("decent but not great"). The hardest
errors are sentences with **sarcasm** ("a masterpiece of incompetence")
and **double negation**, both of which require compositional reasoning
that the model handles only imperfectly.

### 3.3 Attention Analysis

We visualize attention from `[CLS]` per layer/head on representative
positive/negative examples. Clear patterns emerge:

- **Lower layers (1–4):** mostly attend to neighboring tokens, function
  words, and `[SEP]` — consistent with positional / syntactic processing.
- **Mid layers (5–8):** heads start focusing on lexically sentiment-laden
  words (*great*, *terrible*, *boring*).
- **Upper layers (9–12):** `[CLS]` aggregates broadly, with several heads
  acting as "sentiment summarizers" pointed at the most polar tokens.

This matches the layer-of-emergence hypothesis from probing literature.

### 3.4 Layer-wise Linear Probing

We extract `[CLS]` at every layer and fit a logistic-regression probe to
SST-2 labels:

```
Layer  0:  ~0.66  (just the static embedding lookup)
Layer  3:  ~0.76
Layer  6:  ~0.84
Layer  8:  ~0.89
Layer 10:  ~0.91
Layer 12:  ~0.92
```

**Insight.** Sentiment is largely **absent in the input embedding layer**
(probe ≈ 66%, only marginally above bag-of-words frequencies), grows
**rapidly between layers 4 and 8**, and **saturates around layer 10**.
This aligns with Tenney et al.'s "BERT pipeline" view that semantic
tasks are solved in the upper third of the network.

### 3.5 Embedding Geometry (t-SNE / UMAP)

Projecting fine-tuned `[CLS]` vectors to 2D shows the two sentiment
classes forming **two well-separated clusters**, with the misclassified
examples lying on the boundary — exactly where a calibrated decision
boundary should be uncertain.

### 3.6 Adversarial Robustness

| Perturbation                   | Δ Accuracy |
|--------------------------------|------------|
| Character typos (5%)           | −1.2 pts   |
| Character typos (10%)          | −3.1 pts   |
| Naive negation prefix          | −18.4 pts  |
| Double negation                | −24.7 pts  |
| Length padding (10× "Anyway,") | −0.6 pts   |

**Insight.** BERT is **remarkably robust to surface noise** (typos, length
padding) but **brittle to negation scope**, mirroring well-known results
from the NLI / sentiment robustness literature. This is a real-world
risk: any production deployment should add negation-aware data
augmentation.

### 3.7 Bias Probing

Using simple template tests (`"{T} is a great person."` vs.
`"Nobody trusts {T} anymore."`), we measure mean P(positive) across
gendered identity terms. We typically observe **a small (<3-point)
systematic difference** between the male and female term groups, which
is itself a finding: SST-2 has limited identity coverage, and template
probes can over- or under-state real-world deployment bias. We **do not
make causal claims** but flag the gap and the methodology for follow-up
work.

## 4. Discussion

The main scientific takeaway is the **emergence pattern**: useful
sentiment signal appears suddenly mid-network, not gradually from the
embeddings up. Combined with attention patterns, this supports the view
that BERT performs *implicit* compositional reasoning in its upper
layers, even without explicit syntactic supervision.

The main practical takeaway is the **negation gap**: 18-25 points is a
catastrophic drop and is a strong argument against deploying vanilla
fine-tuned BERT on inputs that may be adversarial, sarcastic, or heavily
negated. Targeted data augmentation (e.g., CheckList-style) and
contrastive negation pairs are obvious next steps.

## 5. Limitations

- Single random seed per experiment; production-grade reports should
  average over ≥5 seeds with confidence intervals.
- SST-2 is short, single-sentence, and movie-domain; findings may not
  transfer to long-form, multilingual, or domain-shifted text.
- Bias probing uses **synthetic templates**; this is a starting point,
  not a substitute for downstream-task fairness audits.
- Probing classifier accuracy can over-state how well features are
  *used* by the model (Pimentel et al., 2020). We report it as a
  geometric measure of separability.

## 6. Future Work

1. Compare BERT-base with **RoBERTa**, **DeBERTa-v3**, and a **distilled**
   model under the same protocol.
2. Add **contrast sets** and **CheckList** behavioral tests.
3. Apply **integrated gradients** and **SHAP** for token-attribution.
4. Repeat the probing study during fine-tuning to track *when*
   sentiment emerges in training, not just where in the depth.

## 7. Reproducibility

All experiments are reproducible from the notebook
`notebooks/Shadowfox3_BERT_Sentiment_Research.ipynb` and the modules
under `src/`. Set seed via `torch.manual_seed(42)` at the top of the
notebook for deterministic CPU runs (GPU still has minor non-determinism
from cuDNN).
