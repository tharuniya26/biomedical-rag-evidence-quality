import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/processed/pubmedqa_pqal.csv")

DEV_FILE = Path("data/processed/pubmedqa_dev.csv")
TEST_FILE = Path("data/processed/pubmedqa_test.csv")

RANDOM_SEED = 42
DEV_SIZE = 0.20

print("Loading cleaned PubMedQA dataset...")

df = pd.read_csv(DATA_FILE)

dev_df, test_df = train_test_split(
    df,
    test_size=1 - DEV_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["label"],
)

dev_df = dev_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

dev_df.to_csv(DEV_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\nFULL DATASET")
print("Shape:", df.shape)
print(df["label"].value_counts())

print("\nDEV SET")
print("Shape:", dev_df.shape)
print(dev_df["label"].value_counts())
print(dev_df["label"].value_counts(normalize=True).round(3))

print("\nTEST SET")
print("Shape:", test_df.shape)
print(test_df["label"].value_counts())
print(test_df["label"].value_counts(normalize=True).round(3))

overlap = set(dev_df["pmid"]) & set(test_df["pmid"])

print("\nPMID overlap between DEV and TEST:", len(overlap))

assert len(dev_df) == 200
assert len(test_df) == 800
assert len(overlap) == 0

print("\nSplit checks passed.")
print(f"Saved DEV set to: {DEV_FILE}")
print(f"Saved TEST set to: {TEST_FILE}")