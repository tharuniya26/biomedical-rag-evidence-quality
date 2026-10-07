import re
import pandas as pd
from transformers import pipeline

DEV_FILE = "data/processed/pubmedqa_dev.csv"
OUTPUT_FILE = "results/baseline_llm_smoke_test.csv"

print("Loading development set...")
df = pd.read_csv(DEV_FILE)

# Only test 5 questions for now
sample = df.head(5).copy()

print("Loading small local model...")
generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2",
    device=-1,
)


def extract_label(text):
    text = str(text).lower().strip()

    if re.search(r"\byes\b", text):
        return "yes"

    if re.search(r"\bno\b", text):
        return "no"

    if re.search(r"\bmaybe\b", text):
        return "maybe"

    return "unknown"


results = []

for _, row in sample.iterrows():

    question = row["question"]
    context = row["context"]
    gold = row["label"]

    no_context_prompt = f"""
You are answering a biomedical question.

Answer using exactly one word:
yes, no, or maybe.

Question:
{question}

Answer:
"""

    gold_context_prompt = f"""
You are answering a biomedical question using the evidence provided below.

Answer using exactly one word:
yes, no, or maybe.

Evidence:
{context}

Question:
{question}

Answer:
"""

    no_context_output = generator(
        no_context_prompt,
        max_new_tokens=10,
        do_sample=False,
        return_full_text=False,
    )[0]["generated_text"]

    gold_context_output = generator(
        gold_context_prompt,
        max_new_tokens=10,
        do_sample=False,
        return_full_text=False,
    )[0]["generated_text"]

    no_context_label = extract_label(no_context_output)
    gold_context_label = extract_label(gold_context_output)

    results.append(
        {
            "pmid": row["pmid"],
            "question": question,
            "gold": gold,
            "no_context_raw": no_context_output,
            "no_context_pred": no_context_label,
            "gold_context_raw": gold_context_output,
            "gold_context_pred": gold_context_label,
        }
    )

    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("GOLD:", gold)
    print("NO CONTEXT:", no_context_output, "->", no_context_label)
    print("GOLD CONTEXT:", gold_context_output, "->", gold_context_label)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved results to:")
print(OUTPUT_FILE)