import json
import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/pubmedqa/data/ori_pqal.json")
OUTPUT_FILE = Path("data/processed/pubmedqa_pqal.csv")

print("Loading PubMedQA PQA-L...")

with open(RAW_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Number of records: {len(data)}")

rows = []

for pmid, item in data.items():
    question = item.get("QUESTION", "")
    context = item.get("CONTEXTS", [])
    label = item.get("final_decision", "")
    long_answer = item.get("LONG_ANSWER", "")

    if isinstance(context, list):
        context_text = " ".join(context)
    else:
        context_text = str(context)

    rows.append(
        {
            "pmid": pmid,
            "question": question,
            "context": context_text,
            "label": label,
            "long_answer": long_answer,
        }
    )

df = pd.DataFrame(rows)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst example:")
print("PMID:", df.iloc[0]["pmid"])
print("QUESTION:", df.iloc[0]["question"])
print("LABEL:", df.iloc[0]["label"])
print("CONTEXT:", df.iloc[0]["context"][:500])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")
print(f"Final shape: {df.shape}")