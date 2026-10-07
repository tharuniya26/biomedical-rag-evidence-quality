import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = Path("results/bm25_dev_results.csv")
FIGURE_FILE = Path("figures/bm25_retrieval_performance.png")

df = pd.read_csv(RESULTS_FILE)

recall_at_1 = df["hit_at_1"].mean()
recall_at_5 = df["hit_at_5"].mean()

ranks = df["gold_rank_top5"].dropna()

if len(ranks) > 0:
    mrr_at_5 = (1 / ranks).mean()
else:
    mrr_at_5 = 0.0

metrics = pd.Series(
    {
        "Recall@1": recall_at_1,
        "Recall@5": recall_at_5,
        "MRR@5": mrr_at_5,
    }
)

print("BM25 metrics:")
print(metrics.round(3))

FIGURE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(figsize=(7, 5))
bars = plt.bar(
    metrics.index,
    metrics.values,
)

plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.xlabel("Retrieval Metric")
plt.title("BM25 Retrieval Performance on PubMedQA DEV")

for bar, value in zip(bars, metrics.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
    )

plt.tight_layout()
plt.savefig(
    FIGURE_FILE,
    dpi=300,
)

plt.close()

print("\nSaved figure to:")
print(FIGURE_FILE)