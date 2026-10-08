import pandas as pd
from pathlib import Path

MASTER_FILE = Path(
    "data/processed/pubmedqa_dev_master_conditions.csv"
)

CONTRADICTION_FILE = Path(
    "data/processed/pubmedqa_dev_with_contradiction.csv"
)

OUTPUT_FILE = Path(
    "data/processed/pubmedqa_dev_all_conditions.csv"
)

print("Loading files...")

master_df = pd.read_csv(MASTER_FILE)
contradiction_df = pd.read_csv(CONTRADICTION_FILE)

# Keep only contradictory rows because gold already exists
contradictory_only = contradiction_df[
    contradiction_df["evidence_condition"] == "contradictory"
].copy()

common_columns = [
    "pmid",
    "question",
    "gold_label",
    "evidence_condition",
    "evidence_text",
    "evidence_source_pmid",
]

master_df = master_df[common_columns]
contradictory_only = contradictory_only[common_columns]

all_df = pd.concat(
    [
        master_df,
        contradictory_only,
    ],
    ignore_index=True,
)

print("\nCondition counts:")
print(
    all_df["evidence_condition"]
    .value_counts()
)

print("\nTotal rows:")
print(len(all_df))

print("\nUnique questions:")
print(
    all_df["pmid"].nunique()
)

conditions_per_question = (
    all_df
    .groupby("pmid")["evidence_condition"]
    .nunique()
)

print("\nConditions per question:")
print(
    conditions_per_question
    .value_counts()
)

duplicates = all_df.duplicated(
    subset=[
        "pmid",
        "evidence_condition",
    ]
).sum()

print("\nDuplicate question-condition rows:")
print(duplicates)

assert len(all_df) == 1000
assert all_df["pmid"].nunique() == 200
assert conditions_per_question.min() == 5
assert conditions_per_question.max() == 5
assert duplicates == 0

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

all_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nAll five-condition checks passed.")

print("\nSaved to:")
print(OUTPUT_FILE)