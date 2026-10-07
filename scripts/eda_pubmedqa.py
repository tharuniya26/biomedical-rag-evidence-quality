import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("data/processed/pubmedqa_pqal.csv")
FIGURE_FILE = Path("figures/pubmedqa_label_distribution.png")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nLabel proportions:")
print(df["label"].value_counts(normalize=True).round(3))

print("\nQuestion length summary:")
print(df["question"].str.split().str.len().describe())

print("\nContext length summary:")
print(df["context"].str.split().str.len().describe())

label_counts = df["label"].value_counts().reindex(
    ["yes", "no", "maybe"]
)

FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(7, 5))
label_counts.plot(kind="bar")
plt.title("PubMedQA PQA-L Label Distribution")
plt.xlabel("Gold Label")
plt.ylabel("Number of Questions")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(FIGURE_FILE, dpi=300)
plt.close()

print(f"\nSaved figure to: {FIGURE_FILE}")