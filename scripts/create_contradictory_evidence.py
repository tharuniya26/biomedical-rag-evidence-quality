import pandas as pd
from pathlib import Path

MASTER_FILE = Path(
    "data/processed/pubmedqa_dev_master_conditions.csv"
)

OUTPUT_FILE = Path(
    "data/processed/pubmedqa_dev_with_contradiction.csv"
)

print("Loading master evidence-condition file...")

df = pd.read_csv(MASTER_FILE)

# Keep only the gold rows as the source for contradiction generation
gold_df = df[
    df["evidence_condition"] == "gold"
].copy()

rows = []

for _, row in gold_df.iterrows():

    pmid = str(row["pmid"])
    question = str(row["question"])
    gold_label = str(row["gold_label"])
    gold_evidence = str(row["evidence_text"])

    # --------------------------------------------------
    # Synthetic contradiction
    # --------------------------------------------------
    #
    # We create a controlled contradictory statement
    # based on the gold yes/no label.
    #
    # For "maybe", we do not create a hard yes/no flip;
    # instead we state that the evidence is inconclusive.
    #

    if gold_label == "yes":
        contradiction_text = (
            "The available evidence indicates that the answer "
            "to this question is no. The reported findings do not "
            "support the proposed association or effect."
        )

    elif gold_label == "no":
        contradiction_text = (
            "The available evidence indicates that the answer "
            "to this question is yes. The reported findings support "
            "the proposed association or effect."
        )

    else:
        contradiction_text = (
            "The available evidence provides a definite conclusion "
            "rather than an uncertain or inconclusive result."
        )

    # Add original gold condition
    rows.append(
        {
            "pmid": pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "gold",
            "evidence_text": gold_evidence,
            "evidence_source_pmid": pmid,
        }
    )

    # Add contradictory condition
    rows.append(
        {
            "pmid": pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "contradictory",
            "evidence_text": contradiction_text,
            "evidence_source_pmid": "synthetic",
        }
    )

contradiction_df = pd.DataFrame(rows)

print("\nCondition counts:")
print(
    contradiction_df["evidence_condition"]
    .value_counts()
)

print("\nTotal rows:")
print(len(contradiction_df))

print("\nUnique questions:")
print(
    contradiction_df["pmid"].nunique()
)

print("\nExamples:")

for pmid in contradiction_df["pmid"].unique()[:3]:

    example = contradiction_df[
        contradiction_df["pmid"] == pmid
    ]

    print("\n" + "=" * 70)

    for _, row in example.iterrows():

        print(
            "CONDITION:",
            row["evidence_condition"],
        )

        print(
            "QUESTION:",
            row["question"],
        )

        print(
            "GOLD LABEL:",
            row["gold_label"],
        )

        print(
            "EVIDENCE:",
            row["evidence_text"][:500],
        )

        print()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

contradiction_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved to:")
print(OUTPUT_FILE)