# Biomedical RAG Evidence Quality

This project investigates how evidence quality affects retrieval-augmented generation (RAG) for biomedical question answering using PubMedQA and Qwen2.5-1.5B-Instruct.

## Objective

The aim of the project is to evaluate whether retrieved biomedical evidence consistently improves question answering, and to study how poor-quality evidence affects model accuracy, confidence, and abstention.

## Dataset

The study uses a processed subset of the PubMedQA dataset.

A controlled development set of 200 biomedical questions was evaluated under five evidence conditions:

- Gold evidence
- Missing evidence
- Irrelevant evidence
- Topic-matched irrelevant evidence
- Contradictory evidence

A no-context baseline was also evaluated.

## Retrieval

BM25 was used as the retrieval baseline.

Retrieval performance:

- Recall@1: 0.945
- Recall@5: 0.970
- MRR@5: 0.984

## Language Model

The generative model used was:

Qwen2.5-1.5B-Instruct

Inference was performed using a Tesla T4 GPU in Google Colab.

## Main Results

Accuracy by condition:

| Condition | Accuracy |
|---|---:|
| No context | 45.0% |
| Gold | 62.0% |
| Missing | 40.5% |
| Irrelevant | 33.0% |
| Topic-matched irrelevant | 34.0% |
| Contradictory | 6.5% |

Gold evidence improved model performance compared with the no-context baseline.

Contradictory evidence produced the most severe failure, reducing accuracy to 6.5%.

## Confidence Analysis

Mean model confidence was highest under contradictory evidence:

- Gold: 0.872
- Missing: 0.751
- Irrelevant: 0.747
- Topic-matched irrelevant: 0.741
- Contradictory: 0.909

This shows that the model could become highly confident even when it was incorrect.

## Selective Abstention

Confidence-based abstention was evaluated across several thresholds.

At a confidence threshold of 0.90:

- Gold coverage: 61.5%
- Gold selective accuracy: 69.9%
- Contradictory coverage: 80.0%
- Contradictory selective accuracy: 0%

This indicates that confidence alone was not sufficient to detect misleading evidence.

## Repository Structure

```text
biomedical-rag-evidence-quality/
├── data/
├── figures/
├── notebooks/
├── paper/
├── results/
├── scripts/
├── .gitignore
├── requirements.txt
└── README.md