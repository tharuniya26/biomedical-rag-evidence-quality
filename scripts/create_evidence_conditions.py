import random
import pandas as pd
from pathlib import Path

DEV_FILE = Path("data/processed/pubmedqa_dev.csv")
FULL_FILE = Path("data/processed/pubmedqa_pqal.csv")
OUTPUT_FILE = Path(
    "data/processed/pubmedqa_dev_evidence_conditions.csv"
)

RANDOM_SEED = 42

print("Loading datasets...")

dev_df = pd.read_csv(DEV_FILE)
full_df = pd.read_csv(FULL_FILE)

random.seed(RANDOM_SEED)

rows = []

for _, row in dev_df.iterrows():

    gold_pmid = str(row["pmid"])
    question = str(row["question"])
    gold_label = str(row["label"])
    gold_context = str(row["context"])

    # --------------------------------------------------
    # 1. GOLD EVIDENCE
    # --------------------------------------------------

    rows.append(
        {
            "pmid": gold_pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "gold",
            "evidence_text": gold_context,
            "evidence_source_pmid": gold_pmid,
        }
    )

    # --------------------------------------------------
    # 2. IRRELEVANT EVIDENCE
    # --------------------------------------------------

    candidates = full_df[
        full_df["pmid"].astype(str) != gold_pmid
    ]

    irrelevant_row = candidates.sample(
        n=1,
        random_state=random.randint(
            0,
            1_000_000,
        ),
    ).iloc[0]

    irrelevant_context = str(
        irrelevant_row["context"]
    )

    irrelevant_pmid = str(
        irrelevant_row["pmid"]
    )

    rows.append(
        {
            "pmid": gold_pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "irrelevant",
            "evidence_text": irrelevant_context,
            "evidence_source_pmid": irrelevant_pmid,
        }
    )

    # --------------------------------------------------
    # 3. MISSING / INSUFFICIENT EVIDENCE
    # --------------------------------------------------

    words = gold_context.split()

    # Keep only the first 25% of the context
    keep_words = max(
        1,
        int(len(words) * 0.25),
    )

    missing_context = " ".join(
        words[:keep_words]
    )

    rows.append(
        {
            "pmid": gold_pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "missing",
            "evidence_text": missing_context,
            "evidence_source_pmid": gold_pmid,
        }
    )


conditions_df = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

conditions_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nEvidence condition counts:")
print(
    conditions_df["evidence_condition"]
    .value_counts()
)

print("\nTotal rows:")
print(len(conditions_df))

print("\nUnique questions:")
print(
    conditions_df["pmid"].nunique()
)

print("\nAverage evidence length by condition:")

conditions_df["evidence_words"] = (
    conditions_df["evidence_text"]
    .str.split()
    .str.len()
)

print(
    conditions_df.groupby(
        "evidence_condition"
    )["evidence_words"]
    .mean()
    .round(1)
)

print("\nExample question:")

example_pmid = (
    conditions_df.iloc[0]["pmid"]
)

example_rows = conditions_df[
    conditions_df["pmid"] == example_pmid
]

for _, ex in example_rows.iterrows():

    print("\n" + "=" * 70)

    print(
        "CONDITION:",
        ex["evidence_condition"],
    )

    print(
        "QUESTION:",
        ex["question"],
    )

    print(
        "SOURCE PMID:",
        ex["evidence_source_pmid"],
    )

    print(
        "EVIDENCE PREVIEW:",
        ex["evidence_text"][:500],
    )

print("\nSaved to:")
print(OUTPUT_FILE)