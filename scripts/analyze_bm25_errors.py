import pandas as pd

RESULTS_FILE = "results/bm25_dev_results.csv"

df = pd.read_csv(RESULTS_FILE)

print("Total DEV questions:", len(df))

print("\nOverall retrieval metrics:")
print("Recall@1:", round(df["hit_at_1"].mean(), 3))
print("Recall@5:", round(df["hit_at_5"].mean(), 3))

print("\nRecall@1 by gold label:")
print(
    df.groupby("gold_label")["hit_at_1"]
    .mean()
    .round(3)
)

print("\nRecall@5 by gold label:")
print(
    df.groupby("gold_label")["hit_at_5"]
    .mean()
    .round(3)
)

failures = df[df["hit_at_5"] == 0].copy()

print("\nNumber of Recall@5 failures:")
print(len(failures))

print("\nFailed questions:")
for _, row in failures.iterrows():
    print("\n" + "=" * 70)
    print("PMID:", row["pmid"])
    print("LABEL:", row["gold_label"])
    print("QUESTION:", row["question"])
    print("TOP-5:", row["retrieved_pmids_top5"])

failures.to_csv(
    "results/bm25_retrieval_failures.csv",
    index=False,
)

print("\nSaved failures to:")
print("results/bm25_retrieval_failures.csv")