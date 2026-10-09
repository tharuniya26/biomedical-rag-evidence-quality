import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_FILE = Path(
    "results/qwen25_15b_flip_analysis.csv"
)

OUTPUT_FILE = Path(
    "figures/qwen25_helpful_harmful_flips.png"
)

df = pd.read_csv(INPUT_FILE)

condition_labels = {
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

width = 0.35

fig, ax = plt.subplots(
    figsize=(10, 6)
)

helpful_bars = ax.bar(
    x - width / 2,
    df["helpful_flips"],
    width,
    label="Helpful flips",
)

harmful_bars = ax.bar(
    x + width / 2,
    df["harmful_flips"],
    width,
    label="Harmful flips",
)

ax.set_xlabel(
    "Evidence Condition"
)

ax.set_ylabel(
    "Number of Questions"
)

ax.set_title(
    "Helpful and Harmful Answer Flips Relative to No Context"
)

ax.set_xticks(x)

ax.set_xticklabels(
    df["display_condition"]
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

for bars in [
    helpful_bars,
    harmful_bars,
]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
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

print("\nFlip summary:")
print(
    df[
        [
            "condition",
            "helpful_flips",
            "harmful_flips",
            "net_help",
        ]
    ].to_string(index=False)
)