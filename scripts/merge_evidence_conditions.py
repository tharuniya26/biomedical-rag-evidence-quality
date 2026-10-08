import pandas as pd
from pathlib import Path

BASE_FILE = Path(
    "data/processed/pubmedqa_dev_evidence_conditions.csv"
)

TOPIC_FILE = Path(
    "data/processed/pubmedqa_dev_topic_matched.csv"
)

OUTPUT_FILE = Path(
    "data/processed/pubmedqa_dev_master_conditions.csv"
)

print("Loading evidence-condition files...")

base_df = pd.read_csv(BASE_FILE)
topic_df = pd.read_csv(TOPIC_FILE)

print("\nBase conditions:")
print(
    base_df["evidence_condition"]
    .value_counts()
)

print("\nTopic-matched conditions:")
print(
    topic_df["evidence_condition"]
    .value_counts()
)

# Keep only the topic-matched rows from the second file,
# because gold rows already exist in the base file.
topic_only = topic_df[
    topic_df["evidence_condition"]
    == "topic_matched_irrelevant"
].copy()

common_columns = [
    "pmid",
    "question",
    "gold_label",
    "evidence_condition",
    "evidence_text",
    "evidence_source_pmid",
]

base_df = base_df[common_columns]
topic_only = topic_only[common_columns]

master_df = pd.concat(
    [
        base_df,
        topic_only,
    ],
    ignore_index=True,
)

print("\nMaster condition counts:")
print(
    master_df["evidence_condition"]
    .value_counts()
)

print("\nTotal rows:")
print(len(master_df))

print("\nUnique questions:")
print(
    master_df["pmid"]
    .nunique()
)

conditions_per_question = (
    master_df
    .groupby("pmid")["evidence_condition"]
    .nunique()
)

print("\nConditions per question:")
print(
    conditions_per_question
    .value_counts()
)

duplicates = master_df.duplicated(
    subset=[
        "pmid",
        "evidence_condition",
    ]
).sum()

print("\nDuplicate question-condition rows:")
print(duplicates)

assert len(master_df) == 800
assert master_df["pmid"].nunique() == 200
assert conditions_per_question.min() == 4
assert conditions_per_question.max() == 4
assert duplicates == 0

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

master_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nAll master-condition checks passed.")

print("\nSaved to:")
print(OUTPUT_FILE)