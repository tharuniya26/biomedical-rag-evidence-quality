import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PREDICTIONS_FILE = "results/qwen25_15b_dev_predictions.csv"
NO_CONTEXT_FILE = "results/qwen25_15b_no_context_dev.csv"

print("Loading LLM result files...")

pred_df = pd.read_csv(PREDICTIONS_FILE)
no_context_df = pd.read_csv(NO_CONTEXT_FILE)

LABELS = ["yes", "no", "maybe"]


def print_metrics(name, df):

    y_true = df["gold_label"]
    y_pred = df["prediction"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Accuracy:",
        round(accuracy, 3),
    )

    print(
        "Balanced accuracy:",
        round(balanced_acc, 3),
    )

    print(
        "Macro-F1:",
        round(macro_f1, 3),
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS,
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )


# --------------------------------------------------
# 1. No-context baseline
# --------------------------------------------------

print_metrics(
    "NO CONTEXT",
    no_context_df,
)


# --------------------------------------------------
# 2. Each evidence condition
# --------------------------------------------------

condition_order = [
    "gold",
    "missing",
    "irrelevant",
    "topic_matched_irrelevant",
    "contradictory",
]

for condition in condition_order:

    condition_df = pred_df[
        pred_df["evidence_condition"]
        == condition
    ].copy()

    print_metrics(
        condition.upper(),
        condition_df,
    )


# --------------------------------------------------
# 3. Compact metric table
# --------------------------------------------------

summary_rows = []

no_context_metrics = {
    "condition": "no_context",
    "accuracy": accuracy_score(
        no_context_df["gold_label"],
        no_context_df["prediction"],
    ),
    "balanced_accuracy": balanced_accuracy_score(
        no_context_df["gold_label"],
        no_context_df["prediction"],
    ),
    "macro_f1": f1_score(
        no_context_df["gold_label"],
        no_context_df["prediction"],
        labels=LABELS,
        average="macro",
        zero_division=0,
    ),
}

summary_rows.append(
    no_context_metrics
)

for condition in condition_order:

    cdf = pred_df[
        pred_df["evidence_condition"]
        == condition
    ]

    summary_rows.append(
        {
            "condition": condition,
            "accuracy": accuracy_score(
                cdf["gold_label"],
                cdf["prediction"],
            ),
            "balanced_accuracy": balanced_accuracy_score(
                cdf["gold_label"],
                cdf["prediction"],
            ),
            "macro_f1": f1_score(
                cdf["gold_label"],
                cdf["prediction"],
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),
        }
    )

summary_df = pd.DataFrame(
    summary_rows
)

summary_df[
    [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
] = summary_df[
    [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
].round(3)

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

summary_df.to_csv(
    "results/qwen25_15b_metric_summary.csv",
    index=False,
)


# --------------------------------------------------
# 4. Helpful and harmful flips relative to no context
# --------------------------------------------------

baseline = no_context_df[
    [
        "pmid",
        "gold_label",
        "prediction",
        "correct",
    ]
].copy()

baseline = baseline.rename(
    columns={
        "prediction": "baseline_prediction",
        "correct": "baseline_correct",
    }
)

flip_rows = []

for condition in condition_order:

    cdf = pred_df[
        pred_df["evidence_condition"]
        == condition
    ][
        [
            "pmid",
            "prediction",
            "correct",
        ]
    ].copy()

    cdf = cdf.rename(
        columns={
            "prediction": "condition_prediction",
            "correct": "condition_correct",
        }
    )

    merged = baseline.merge(
        cdf,
        on="pmid",
        how="inner",
    )

    helpful = (
        (merged["baseline_correct"] == 0)
        &
        (merged["condition_correct"] == 1)
    ).sum()

    harmful = (
        (merged["baseline_correct"] == 1)
        &
        (merged["condition_correct"] == 0)
    ).sum()

    unchanged_correct = (
        (merged["baseline_correct"] == 1)
        &
        (merged["condition_correct"] == 1)
    ).sum()

    unchanged_wrong = (
        (merged["baseline_correct"] == 0)
        &
        (merged["condition_correct"] == 0)
    ).sum()

    flip_rows.append(
        {
            "condition": condition,
            "helpful_flips": helpful,
            "harmful_flips": harmful,
            "unchanged_correct": unchanged_correct,
            "unchanged_wrong": unchanged_wrong,
            "net_help": helpful - harmful,
        }
    )

flip_df = pd.DataFrame(
    flip_rows
)

print("\n" + "=" * 70)
print("ANSWER FLIPS VS NO CONTEXT")
print("=" * 70)

print(
    flip_df.to_string(
        index=False
    )
)

flip_df.to_csv(
    "results/qwen25_15b_flip_analysis.csv",
    index=False,
)

print("\nSaved:")
print("results/qwen25_15b_metric_summary.csv")
print("results/qwen25_15b_flip_analysis.csv")