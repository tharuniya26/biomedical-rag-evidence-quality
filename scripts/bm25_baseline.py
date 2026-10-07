import re
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

DEV_FILE = Path("data/processed/pubmedqa_dev.csv")
FULL_FILE = Path("data/processed/pubmedqa_pqal.csv")
OUTPUT_FILE = Path("results/bm25_dev_results.csv")

TOP_K = 5


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


print("Loading PubMedQA datasets...")

dev_df = pd.read_csv(DEV_FILE)
full_df = pd.read_csv(FULL_FILE)

print(f"DEV questions: {len(dev_df)}")
print(f"Retrieval corpus size: {len(full_df)}")

print("\nBuilding BM25 corpus...")

corpus_texts = full_df["context"].fillna("").tolist()
tokenized_corpus = [tokenize(text) for text in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index built.")

results = []

for idx, row in dev_df.iterrows():

    question = row["question"]
    gold_pmid = str(row["pmid"])
    gold_label = row["label"]

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:TOP_K]

    retrieved_pmids = [
        str(full_df.iloc[i]["pmid"])
        for i in ranked_indices
    ]

    retrieved_scores = [
        float(scores[i])
        for i in ranked_indices
    ]

    hit_at_1 = int(
        gold_pmid == retrieved_pmids[0]
    )

    hit_at_5 = int(
        gold_pmid in retrieved_pmids
    )

    rank = None

    for position, pmid in enumerate(retrieved_pmids, start=1):
        if pmid == gold_pmid:
            rank = position
            break

    results.append(
        {
            "pmid": gold_pmid,
            "question": question,
            "gold_label": gold_label,
            "retrieved_pmid_1": retrieved_pmids[0],
            "retrieved_score_1": retrieved_scores[0],
            "retrieved_pmids_top5": "|".join(retrieved_pmids),
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "gold_rank_top5": rank,
        }
    )

    if idx < 5:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("GOLD PMID:", gold_pmid)
        print("TOP-5 RETRIEVED PMIDs:", retrieved_pmids)
        print("HIT@1:", hit_at_1)
        print("HIT@5:", hit_at_5)

results_df = pd.DataFrame(results)

recall_at_1 = results_df["hit_at_1"].mean()
recall_at_5 = results_df["hit_at_5"].mean()

ranks = results_df["gold_rank_top5"].dropna()

if len(ranks) > 0:
    mrr_at_5 = (1 / ranks).mean()
else:
    mrr_at_5 = 0.0

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("BM25 RETRIEVAL RESULTS")
print("=" * 70)

print(f"Recall@1: {recall_at_1:.3f}")
print(f"Recall@5: {recall_at_5:.3f}")
print(f"MRR@5:    {mrr_at_5:.3f}")

print("\nSaved results to:")
print(OUTPUT_FILE)