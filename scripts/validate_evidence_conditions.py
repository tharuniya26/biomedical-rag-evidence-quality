import pandas as pd

FILE = "data/processed/pubmedqa_dev_evidence_conditions.csv"

df = pd.read_csv(FILE)

print("Dataset shape:", df.shape)

print("\nCondition counts:")
print(df["evidence_condition"].value_counts())

print("\nConditions per question:")
counts = df.groupby("pmid")["evidence_condition"].nunique()
print(counts.value_counts())

print("\nQuestions missing one or more conditions:")
bad = counts[counts != 3]
print(len(bad))

print("\nChecking irrelevant-source leakage...")

irrelevant = df[df["evidence_condition"] == "irrelevant"].copy()

same_source = (
    irrelevant["pmid"].astype(str)
    == irrelevant["evidence_source_pmid"].astype(str)
)

print("Irrelevant rows using same PMID as question:")
print(same_source.sum())

print("\nDuplicate rows:")
print(
    df.duplicated(
        subset=[
            "pmid",
            "evidence_condition"
        ]
    ).sum()
)

assert len(df) == 600
assert bad.empty
assert same_source.sum() == 0

print("\nAll evidence-condition checks passed.")