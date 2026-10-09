import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CONF_FILE = Path(
    "results/qwen25_15b_confidence_results.csv"
)

ABST_FILE = Path(
    "results/qwen25_15b_abstention_overall.csv"
)

COND_ABST_FILE = Path(
    "results/qwen25_15b_abstention_by_condition.csv"
)

CONF_FIG = Path(
    "figures/qwen25_confidence_by_condition.png"
)

ABST_FIG = Path(
    "figures/qwen25_coverage_selective_accuracy.png"
)

# --------------------------------------------------
# 1. Confidence by evidence condition
# --------------------------------------------------

conf_df = pd.read_csv(CONF_FILE)

condition_order = [
    "gold",
    "missing",
    "irrelevant",
    "topic_matched_irrelevant",
    "contradictory",
]

condition_labels = {
    "gold": "Gold",
    "missing": "Missing",
    "irrelevant": "Irrelevant",
    "topic_matched_irrelevant": "Topic-Matched\nIrrelevant",
    "contradictory": "Contradictory",
}

mean_conf = (
    conf_df
    .groupby("evidence_condition")["confidence"]
    .mean()
    .reindex(condition_order)
)

xlabels = [
    condition_labels[c]
    for c in condition_order
]

plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    xlabels,
    mean_conf.values,
)

plt.ylabel(
    "Mean Confidence"
)

plt.xlabel(
    "Evidence Condition"
)

plt.title(
    "Model Confidence Across Evidence Conditions"
)

plt.ylim(
    0,
    1.0,
)

plt.grid(
    axis="y",
    alpha=0.25,
)

for bar, value in zip(
    bars,
    mean_conf.values,
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.015,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()

CONF_FIG.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    CONF_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# --------------------------------------------------
# 2. Coverage vs selective accuracy
# --------------------------------------------------

abst_df = pd.read_csv(
    ABST_FILE
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    abst_df["coverage"],
    abst_df["selective_accuracy"],
    marker="o",
)

for _, row in abst_df.iterrows():

    plt.text(
        row["coverage"],
        row["selective_accuracy"] + 0.01,
        f'{row["threshold"]:.2f}',
        ha="center",
        fontsize=8,
    )

plt.xlabel(
    "Coverage"
)

plt.ylabel(
    "Selective Accuracy"
)

plt.title(
    "Coverage vs Selective Accuracy"
)

plt.xlim(
    0,
    1.05,
)

plt.ylim(
    0,
    1.0,
)

plt.grid(
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    ABST_FIG,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("Saved:")
print(CONF_FIG)
print(ABST_FIG)

# --------------------------------------------------
# 3. Quick condition-specific summary
# --------------------------------------------------

cond_abst_df = pd.read_csv(
    COND_ABST_FILE
)

print("\nCondition-specific abstention at threshold 0.90:")

print(
    cond_abst_df[
        cond_abst_df["threshold"] == 0.90
    ][
        [
            "condition",
            "coverage",
            "selective_accuracy",
        ]
    ].to_string(index=False)
)