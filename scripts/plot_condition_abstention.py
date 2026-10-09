import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_FILE = Path(
    "results/qwen25_15b_abstention_by_condition.csv"
)

OUTPUT_FILE = Path(
    "figures/qwen25_condition_abstention_curves.png"
)

df = pd.read_csv(INPUT_FILE)

conditions = [
    "gold",
    "missing",
    "irrelevant",
    "topic_matched_irrelevant",
    "contradictory",
]

labels = {
    "gold": "Gold",
    "missing": "Missing",
    "irrelevant": "Irrelevant",
    "topic_matched_irrelevant": "Topic-Matched Irrelevant",
    "contradictory": "Contradictory",
}

plt.figure(figsize=(9, 6))

for condition in conditions:

    cdf = df[
        df["condition"] == condition
    ].sort_values("coverage")

    plt.plot(
        cdf["coverage"],
        cdf["selective_accuracy"],
        marker="o",
        label=labels[condition],
    )

plt.xlabel("Coverage")
plt.ylabel("Selective Accuracy")

plt.title(
    "Selective Accuracy vs Coverage by Evidence Condition"
)

plt.xlim(0, 1.05)
plt.ylim(0, 1.0)

plt.grid(alpha=0.25)

plt.legend()

plt.tight_layout()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved figure to:")
print(OUTPUT_FILE)