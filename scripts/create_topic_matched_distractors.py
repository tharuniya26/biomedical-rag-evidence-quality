import re
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi

DEV_FILE = Path("data/processed/pubmedqa_dev.csv")
FULL_FILE = Path("data/processed/pubmedqa_pqal.csv")

OUTPUT_FILE = Path(
    "data/processed/pubmedqa_dev_topic_matched.csv"
)


def tokenize(text):
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


print("Loading datasets...")

dev_df = pd.read_csv(DEV_FILE)
full_df = pd.read_csv(FULL_FILE)

print("Building BM25 index...")

corpus_texts = (
    full_df["context"]
    .fillna("")
    .tolist()
)

tokenized_corpus = [
    tokenize(text)
    for text in corpus_texts
]

bm25 = BM25Okapi(
    tokenized_corpus
)

rows = []

for _, row in dev_df.iterrows():

    question = str(row["question"])
    gold_pmid = str(row["pmid"])
    gold_label = str(row["label"])
    gold_context = str(row["context"])

    query_tokens = tokenize(
        question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = (
        scores.argsort()[::-1]
    )

    distractor_row = None
    distractor_score = None

    for idx in ranked_indices:

        candidate_pmid = str(
            full_df.iloc[idx]["pmid"]
        )

        if candidate_pmid != gold_pmid:

            distractor_row = (
                full_df.iloc[idx]
            )

            distractor_score = float(
                scores[idx]
            )

            break

    if distractor_row is None:
        raise RuntimeError(
            f"No distractor found for PMID {gold_pmid}"
        )

    rows.append(
        {
            "pmid": gold_pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "gold",
            "evidence_text": gold_context,
            "evidence_source_pmid": gold_pmid,
            "bm25_score": None,
        }
    )

    rows.append(
        {
            "pmid": gold_pmid,
            "question": question,
            "gold_label": gold_label,
            "evidence_condition": "topic_matched_irrelevant",
            "evidence_text": str(
                distractor_row["context"]
            ),
            "evidence_source_pmid": str(
                distractor_row["pmid"]
            ),
            "bm25_score": distractor_score,
        }
    )

conditions_df = pd.DataFrame(
    rows
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

conditions_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nCondition counts:")
print(
    conditions_df[
        "evidence_condition"
    ].value_counts()
)

print("\nTotal rows:")
print(len(conditions_df))

print("\nUnique questions:")
print(
    conditions_df[
        "pmid"
    ].nunique()
)

topic_rows = conditions_df[
    conditions_df[
        "evidence_condition"
    ]
    == "topic_matched_irrelevant"
]

same_source = (
    topic_rows["pmid"].astype(str)
    ==
    topic_rows[
        "evidence_source_pmid"
    ].astype(str)
)

print("\nSame PMID leakage:")
print(
    same_source.sum()
)

print(
    "\nAverage BM25 score "
    "for topic-matched distractors:"
)

print(
    round(
        topic_rows[
            "bm25_score"
        ].mean(),
        3
    )
)

print("\nExample:")

example_pmid = (
    conditions_df.iloc[0][
        "pmid"
    ]
)

example_rows = (
    conditions_df[
        conditions_df["pmid"]
        == example_pmid
    ]
)

for _, ex in (
    example_rows.iterrows()
):

    print("\n" + "=" * 70)

    print(
        "CONDITION:",
        ex[
            "evidence_condition"
        ],
    )

    print(
        "QUESTION:",
        ex["question"],
    )

    print(
        "SOURCE PMID:",
        ex[
            "evidence_source_pmid"
        ],
    )

    print(
        "EVIDENCE PREVIEW:",
        ex["evidence_text"][:500],
    )

print("\nSaved to:")
print(OUTPUT_FILE)