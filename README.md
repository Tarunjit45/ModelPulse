# 🩺 ModelPulse — LLM Drift Detection & Reliability Monitor

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Typer](https://img.shields.io/badge/CLI-Typer-009688?style=for-the-badge)](https://typer.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Inference-Ollama-000000?style=for-the-badge)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**ModelPulse** is a lightweight, local-first monitoring engine and CLI tool designed to detect **silent degradation, semantic drift, tone shift, and verbosity drift** in production Large Language Models.

---

## 📌 The Problem: Silent LLM Degradation

When software APIs break, they throw 500 errors. When LLMs degrade—due to prompt tweaks, model version deprecations, temperature changes, or provider quantization—**they don't throw errors: they silently drift**.
- Tone shifts from professional to casual or sycophantic.
- Output length balloons or contracts dramatically.
- Semantic meaning diverges from established domain baselines.
- Critical safety keywords or instructions disappear.

**ModelPulse provides early warning signals** by establishing empirical behavioral baselines and scoring model drift across multiple dimensions.

---

## ⚙️ Core Architecture & Drift Analyzers

```
                           +---------------------------+
                           |  Baseline Configuration   |
                           +---------------------------+
                                         |
                                         v
+------------------+         +-----------------------+         +--------------------+
| Prompt Suite     | ------> |      ModelPulse       | <------ | Production Output  |
| (.yaml / JSON)   |         |      CLI Engine       |         | (Live / Ollama)    |
+------------------+         +-----------------------+         +--------------------+
                                         |
              +--------------------------+--------------------------+
              |                          |                          |
              v                          v                          v
    [ Semantic Drift ]            [ Tone Drift ]            [ Length Drift ]
  Cosine similarity via        Affective sentiment &       Token / character
  Sentence Transformers        formality classifier        count penalties
              |                          |                          |
              +--------------------------+--------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Composite Health Score      |
                         |   (0 - 100 Reliability Index) |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Markdown & JSON Reports     |
                         +-------------------------------+
```

### Drift Detection Modules:
1. **Semantic Drift (`modelpulse/drift/semantic.py`):** Uses Sentence Transformers (`all-MiniLM-L6-v2`) to compute high-dimensional embedding vectors and cosine distance between baseline and current outputs.
2. **Tone Drift (`modelpulse/drift/tone.py`):** Analyzes conversational sentiment and formality changes against expected baseline personas.
3. **Length Drift (`modelpulse/drift/length.py`):** Calculates statistical length drift penalties to catch verbosity bloat or truncated responses.
4. **Keyword Drift (`modelpulse/drift/keywords.py`):** Extracts mandatory domain keywords and flags missing technical nomenclature.
5. **Health Scoring (`modelpulse/scoring.py`):** Aggregates weighted drift penalties into a unified reliability health score.

---

## 📁 Repository Layout

```text
ModelPulse/
├── modelpulse/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py             # Typer CLI application entry point
│   ├── baseline.py        # Baseline snapshot capture and schema
│   ├── runner.py          # LLM execution harness (Ollama & Mock runners)
│   ├── scoring.py         # Multi-metric composite health scoring
│   ├── report.py          # Markdown & JSON report generator
│   └── drift/             # Specialized drift detection algorithms
│       ├── __init__.py
│       ├── semantic.py    # Embedding-based cosine similarity analyzer
│       ├── tone.py        # Tone and style shift evaluator
│       ├── length.py      # Response length & verbosity penalty engine
│       └── keywords.py    # Key terminology retention validator
├── prompts/               # Prompt suites for testing
├── pyproject.toml         # Package dependencies & CLI entry point
├── LICENSE                # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install with `pip`:

```bash
git clone https://github.com/Tarunjit45/ModelPulse.git
cd ModelPulse

pip install -e .
```

*Requirements: Python 3.10+, Ollama (for local inference), Sentence Transformers.*

### 2. Initialize a Project
Set up the ModelPulse directory structure:

```bash
modelpulse init
```
This generates the `baselines/`, `prompts/`, and `runs/` directories.

### 3. Capture a Baseline
Record baseline responses from your reference model:

```bash
modelpulse baseline --model llama3.2:1b --prompts prompts/test_suite.json
```

### 4. Run Drift Audits & Generate Reports
Audit the latest model responses against the established baseline:

```bash
modelpulse run --baseline baselines/baseline_v1.json
```

View the generated health score and degradation breakdown in terminal or export as markdown.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
