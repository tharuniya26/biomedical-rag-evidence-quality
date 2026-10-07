from datasets import load_dataset

print("Loading PubMedQA...")

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_labeled"
)

print("\nDataset structure:")
print(dataset)

print("\nFirst example:")
print(dataset["train"][0])

print("\nNumber of examples:")
print(len(dataset["train"]))

print("\nAvailable fields:")
print(dataset["train"].column_names)