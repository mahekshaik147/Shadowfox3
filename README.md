# Shadowfox3 — Advanced NLP Research with BERT

> **A research-oriented, notebook-driven study of BERT for sentiment analysis.**
> Fine-tuning · Attention analysis · Probing · Bias & robustness · Ablations

This repository accompanies an advanced ML/NLP project that goes beyond
"load-a-model-and-predict". It studies *how* and *why* a transformer-based
Language Model (BERT) makes the decisions it does on a sentiment-classification
task, using rigorous experimentation, visualization, and analysis.

---

## 🎯 Project Objectives

1. **Fine-tune `bert-base-uncased`** on the Stanford Sentiment Treebank (SST-2).
2. **Compare** the fine-tuned model against classical baselines
   (Logistic Regression on TF-IDF, BiLSTM with GloVe).
3. **Open the black box** with attention visualizations, layer-wise probing,
   and CLS-embedding geometry.
4. **Stress-test** the model with adversarial perturbations, negation handling,
   and length-bias studies.
5. **Probe for bias** along gender / sentiment-laden identity terms.
6. **Distill insights** into a written research report (`reports/REPORT.md`).

---

## 📁 Repository Layout

```
Shadowfox3/
├── README.md
├── requirements.txt
├── notebooks/
│   └── Shadowfox3_BERT_Sentiment_Research.ipynb   ← main deliverable
├── src/
│   ├── data.py            # dataset loading, preprocessing
│   ├── models.py          # BERT classifier, baselines
│   ├── train.py           # training loop, eval metrics
│   ├── attention.py       # attention extraction & plots
│   ├── probing.py         # layer-wise linear probes
│   ├── robustness.py      # adversarial perturbations
│   └── bias.py            # template-based bias tests
├── reports/
│   └── REPORT.md          # research write-up with findings
├── figures/               # (generated) plots from the notebook
└── data/                  # (auto-downloaded) cached datasets
```

---

## 🚀 Quick Start

🚀 Quick Start

## 1. Environment

Python ≥ 3.9 recommended. GPU strongly recommended for fine-tuning (Google Colab Free T4 is sufficient — the notebook detects GPU automatically).

Install dependencies:

```
pip install -r requirements.txt
```

---

## 2. Run the Notebook (Local System)

Navigate to the project folder:

```
cd Shadowfox3
```

Start Jupyter Notebook:

```
python -m notebook
```

This will open a browser window. Then:

* Go to `notebooks/`
* Open `Shadowfox3_BERT_Sentiment_Research.ipynb`
* Click **Run → Run All Cells**

⚠️ Note:

* On CPU, training can be slow. Set `EPOCHS = 1` in the notebook for quick testing.

---

## 🚀 Run on Google Colab (Recommended)

This project runs best on Google Colab with GPU support.

---


1. Environment
Google Colab is recommended for this project. GPU acceleration (T4) is sufficient for fine-tuning and is automatically detected in the notebook.

---

2. Open the Project in Colab

Go to:
https://colab.research.google.com/

Create a new notebook and run:

```python
!git clone https://github.com/mahekshaik147/Shadowfox3.git
%cd Shadowfox3
Install Dependencies
!pip install -r requirements.txt
Open the Notebook

Click the 📁 (file icon) on the left sidebar
Navigate to:
Shadowfox3/notebooks/Shadowfox3_BERT_Sentiment_Research.ipynb

Make Notebook Editable

Click:

File → Save a copy in Drive

Fix Import Path (IMPORTANT)

Add this at the top of the notebook:

import sys, os
sys.path.append("/content/Shadowfox3")
Enable GPU

Go to:

Runtime → Change runtime type → T4 GPU

Run the Notebook

Click:

Runtime → Run all

## 4. Use Modules Independently

You can also use project modules directly:

```python
from src.data import load_sst2
from src.models import BertSentimentClassifier
from src.train import train_model

train_loader, val_loader, _ = load_sst2(batch_size=32)
model = BertSentimentClassifier()
train_model(model, train_loader, val_loader, epochs=3)
```

---

## ⚠️ Troubleshooting

* If `jupyter` command is not recognized:

  ```
  python -m notebook
  ```

* If training is slow:

  * Reduce epochs
  * Use Google Colab GPU

---


## 🧪 Experiments Included

| #  | Experiment                              | What it shows                                |
|----|-----------------------------------------|----------------------------------------------|
| 1  | Baselines: LogReg(TF-IDF) & BiLSTM      | Lower-bound performance reference            |
| 2  | BERT fine-tuning on SST-2               | The headline model                           |
| 3  | Confusion matrix & error analysis       | Where BERT actually fails                    |
| 4  | Attention head visualization            | What each head attends to                    |
| 5  | Layer-wise CLS-probing                  | Where sentiment "emerges" in the stack       |
| 6  | t-SNE / UMAP of CLS embeddings          | Semantic geometry of representations         |
| 7  | Adversarial robustness (typos, negation)| Brittleness to input perturbations           |
| 8  | Length & position bias study            | Spurious correlations the model exploits     |
| 9  | Template-based bias probing             | Sentiment skew over identity terms           |

---

## 📊 Key Findings (preview)

See `reports/REPORT.md` for the full write-up. Highlights:

- BERT-base reaches **~92% accuracy** on SST-2 dev with only 3 epochs of
  fine-tuning, **+8 points over BiLSTM** and **+12 over TF-IDF + LogReg**.
- Sentiment information is **mostly absent in layers 0–3**, emerges sharply
  around **layers 6–8**, and saturates at layer 10 — consistent with prior
  probing literature (Tenney et al., 2019).
- The model is **highly sensitive to negation scope** but degrades
  gracefully under character-level typos (~3-point drop at 10% noise).
- We detect a **mild positive-sentiment skew** for some identity terms in
  template prompts — discussed in the bias section with caveats.

---

## 📚 References

- Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding.* NAACL 2019.
- Tenney, Das, Pavlick. *BERT Rediscovers the Classical NLP Pipeline.*
  ACL 2019.
- Clark et al. *What Does BERT Look At? An Analysis of BERT's Attention.*
  BlackboxNLP 2019.
- Socher et al. *Recursive Deep Models for Semantic Compositionality Over a
  Sentiment Treebank.* EMNLP 2013.

---

## 📝 License

MIT — see `LICENSE` (add your preferred license file before publishing).

---

*Built as part of the Shadowfox AI/ML research track.*
