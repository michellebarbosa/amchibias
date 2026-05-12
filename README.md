# 🥥 Goan Stereotypes & Low-Resource Bias Evaluation

![Status](https://img.shields.io/badge/Status-Data_Collection-orange)
![Language](https://img.shields.io/badge/Language-Python_|_Konkani_|_English-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview
This research investigates sociolinguistic biases in Large Language Models (LLMs) with a specific focus on the **Goan community** and the **Konkani language**. Unlike standard bias evaluations that focus on broad categories (e.g., gender/race in the US), this project creates a novel, culturally grounded dataset to test how global and local models perceive specific regional identities.

**Core Research Goals:**
1. **Create** the first dataset of stereotypical tuples for the Goan sociolinguistic landscape.
2. **Evaluate** the "Subgroup Performance Gap" across high-resource (English) and low-resource (Konkani) settings.
3. **Compare** Western-centric decoders (Llama-3, Mistral) against regionally optimized encoders (IndicBERT).

---

## Experimental Design

### A) Data Collection (Dataset Creation)
We are curating a "Gold Standard" validation set grounded in local sociological reality.

| Variable | Value | Description |
| :--- | :--- | :--- |
| **Unique Axes** | 8 | Caste, Language, Occupation, Religion, **Nativity**, Region, Age, Gender |
| **Identity Groups** | ~30 | Total subgroups across all axes |
| **Stereotypes** | 5 | Distinct tropes per identity group |
| **Seed Tuples** | 150 | Pre-annotation list |
| **Target Judges** | 3 | **Goan Insiders** (Native speakers for ground truth) |
| **Control Judges** | 3 | **Non-Goan Indians** (Outsiders for perceptual contrast) |
| **Total Judgments** | **900** | $(150 \text{ Tuples} \times 6 \text{ Annotators})$ |

> **Note on Nativity:** "Nativity" in Goa is a distinct axis involving three groups: *Locals*, *Migrant Laborers*, and *Tourists*. This project explicitly models stereotypes between these groups.

### B) Data Modelling (Bias Evaluation)
We test both understanding (Encoder) and generation (Decoder) capabilities.

| Variable | Value | Details |
| :--- | :--- | :--- |
| **Languages** | 2 | English, Konkani |
| **Models** | 4 | **IndicBERT v3** (Encoder), **Llama-3**, **Mistral**, **mGPT** (Decoders) |
| **Prompts** | 5 | Robustness templates (e.g., *"People from [Region] are usually..."*) |
| **Polarity** | 2 | Stereotype vs. Anti-Stereotype pairs |
| **Total Predictions** | **6,000** | $(150 \times 2 \times 5 \times 4)$ |

---

## Tech Stack
* **HuggingFace Transformers**: Pipeline orchestration.
* **PyTorch**: Backend tensor computations.
* **Scikit-learn**: Calculation of performance metrics (F1, Accuracy, MAT).

## Repository Structure
```bash
goan-stereotypes-nlp/
├── data/
│   ├── raw/          # Initial list of 150 tuples
│   └── processed/    # Annotated & Validated JSONs
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── 02_model_inference.ipynb
├── src/              # Helper scripts for metrics
└── README.md         # Project documentation
