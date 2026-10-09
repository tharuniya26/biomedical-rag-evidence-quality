import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_FILE = Path(
    "results/qwen25_15b_metric_summary.csv"
)

OUTPUT_FILE = Path(
    "figures/qwen25_evidence_condition_performance.png"
)

df = pd.read_csv(INPUT_FILE)

condition_labels = {
    "no_context": "No Context",
    "gold": "Gold",
    "missing": "Missing",
    "irrelevant": "Irrelevant",
    "topic_matched_irrelevant": "Topic-Matched\nIrrelevant",
    "contradictory": "Contradictory",
}

df["display_condition"] = (
    df["condition"]
    .map(condition_labels)
)

x = np.arange(len(df))

width = 0.25

fig, ax = plt.subplots(
    figsize=(11, 6)
)

bars1 = ax.bar(
    x - width,
    df["accuracy"],
    width,
    label="Accuracy",
)

bars2 = ax.bar(
    x,
    df["balanced_accuracy"],
    width,
    label="Balanced Accuracy",
)

bars3 = ax.bar(
    x + width,
    df["macro_f1"],
    width,
    label="Macro-F1",
)

ax.set_ylabel("Score")
ax.set_xlabel("Evidence Condition")

ax.set_title(
    "Qwen2.5-1.5B Performance Across Evidence Conditions"
)

ax.set_xticks(x)

ax.set_xticklabels(
    df["display_condition"]
)

ax.set_ylim(
    0,
    0.7,
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

for bars in [
    bars1,
    bars2,
    bars3,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

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