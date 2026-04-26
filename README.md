# 🧬 Explainable AI Framework for Protein Mutation Analysis

> An end-to-end deep learning system for predicting the impact of protein mutations, combining multi-task learning, explainability, and clinical report generation.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

Protein mutations can alter structure, disrupt function, and lead to disease. This project builds an interpretable AI pipeline that not only predicts mutation impact but also explains *why* it happens and translates results into clinical-style reports.

---

## 🔍 What This System Does

| Task | Description |
|------|-------------|
| ✅ Pathogenicity | Classifies mutations as Benign vs Pathogenic |
| 🧬 Disease Category | Multi-class disease classification |
| 🧪 Stability Change | Estimates ΔΔG (protein stability change) |
| 🧠 Explainability | Feature attribution via Integrated Gradients |
| 📄 Clinical Reports | Human-readable reports via local LLM |

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Feature Extraction | ESM-2 (Meta AI) |
| Deep Learning | PyTorch |
| Explainability | Captum (Integrated Gradients) |
| LLM | Ollama (`phi` / `mistral`) |
| Data | ClinVar, FireProtDB |
| Language | Python |

---

## 🏗️ Methodology

```
Mutation Dataset (ClinVar + FireProtDB)
            ↓
     Data Preprocessing
            ↓
     Protein Sequence Input
            ↓
   ESM-2 Feature Extraction
            ↓
 Feature Construction (960D)
            ↓
 Multi-task Neural Network
            ↓
  Explainability (Captum)
            ↓
 Structured Interpretation
            ↓
 Clinical Report (LLM)
```

---

## 📊 Results

| Task | Performance |
|------|-------------|
| Pathogenicity | 80.3% Accuracy |
| Disease Classification | 63.7% Accuracy |
| ΔΔG Prediction | MAE: 0.599, r ≈ 0.81 |

> **🔬 Key Insight:** Mutation-site features contribute ~56% to predictions, aligning with biological expectations.

---

## 🧬 Feature Engineering

We construct a **960-dimensional feature vector** from ESM-2 embeddings:

| Dimension | Description |
|-----------|-------------|
| 🔹 0–319 | Global Protein Context |
| 🔹 320–639 | Mutation-Site Representation |
| 🔹 640–959 | Local Structural Context |

---

## 🧠 Explainability

Using **Captum Integrated Gradients**, we quantify feature importance:

```
Mutation Site   ████████████████████  ~56%  (highest)
Local Context   ████████████          moderate
Global Context  ████                  least
```

This ensures biologically meaningful interpretation.

---

## 📄 Clinical Report Generation

Structured outputs are passed to a local LLM (Ollama) to generate:

- Clinical-style mutation interpretation
- Mechanistic reasoning
- Human-readable insights

All processing happens **fully offline** — no external API dependency.

---

## 🖥️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/protein-mutation-ai.git
cd protein-mutation-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama (for LLM report generation)

Download from: [https://ollama.com](https://ollama.com), then pull a lightweight model:

```bash
ollama pull phi
```

---

## ▶️ Usage

```bash
python Explanation.py
```

This will:

1. Load the trained model
2. Predict mutation impact
3. Generate explainability scores
4. Produce a clinical report

---

## 📂 Project Structure

```
protein-mutation-ai/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── embeddings/
│   ├── train_embeddings.pt
│   ├── val_embeddings.pt
│   └── test_embeddings.pt
│
├── models/
│   └── best_multitask_model.pt
│
├── llm_report.py
├── Explanation.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 📈 Future Improvements

- [ ] Improve disease classification (class imbalance handling)
- [ ] Add mutation-difference embeddings (WT vs MUT)
- [ ] Integrate structural features (AlphaFold)
- [ ] Build web UI / API deployment
- [ ] Add ACMG classification support

---

## ⭐ Acknowledgements

- [Meta AI — ESM-2](https://github.com/facebookresearch/esm)
- [ClinVar Database](https://www.ncbi.nlm.nih.gov/clinvar/)
- [FireProtDB](https://loschmidt.chemi.muni.cz/fireprotdb/)
- [Captum (PyTorch Explainability)](https://captum.ai/)
- [Ollama](https://ollama.com)

---

## 📜 License

This project is licensed under the **MIT License** — see below for details.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
