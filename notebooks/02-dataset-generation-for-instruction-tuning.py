# %%
import glob
import os
from datetime import datetime, timedelta

# Get the current time
now = datetime.now()

# Calculate the time 24 hours ago from the current time
twenty_four_hours_ago = now  # - timedelta(hours=24)

# Pattern to match the files with varying numbers in the file path
file_pattern1 = "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/gemini_categorized_prompts/outputs/*.json"
# file_pattern2 = "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/chatgpt/gemma_7b_it_rewrites_*.json"

# Use glob to find all file paths that match the patterns
file_paths = glob.glob(file_pattern1)
# file_paths2 = glob.glob(file_pattern2)
# file_paths = file_paths1 + file_paths2

# Output the list of recent file paths
print("Files not modified in the last 24 hours:")
for file_path in file_paths:
    print(file_path)

# %%
import json
from datasets import Dataset
import re

# Initialize an empty list to store the extracted entries
entries = []

# Iterate over the recent file paths

# %%
import pandas as pd

df = pd.read_csv(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/gemini_categorized_prompts/prompts.csv",
    delimiter=";",
).set_index("Prompt")

for file_path in file_paths:
    # Open the JSON file and load its contents
    with open(file_path, "r") as file:
        file_contents = file.read()
        entries_raw = re.findall(r"\{.*?\}", file_contents, re.DOTALL)
        for entry_raw in entries_raw:
            try:
                entry = json.loads(entry_raw)

                if isinstance(entry["rewrite_prompt"], float):
                    continue

                # Todo This needs to capture all instances!
                patterns = [
                    r"^.*?[hH]ere is.*?\n\n",
                    r"^.*?[hH]ere \'s.*?\n\n",
                    r"^.*?[Ss]ure, here.*?\n\n",
                    r"^.*?[Ss]ure, here \'s.*?\n\n",
                    r"^.*?[Ss]ure, [Hh]ere*?:",
                ]

                filtered = entry["rewritten_text"]
                for pattern in patterns:
                    filtered = re.sub(pattern, "", filtered, flags=re.DOTALL)

                try:
                    extracted_entry = {
                        "rewrite_prompt": entry["rewrite_prompt"],
                        "original_text": entry["original_text"],
                        "rewritten_text": filtered.strip('"'),
                        "category": df.loc[entry["rewrite_prompt"], "Category"]
                        if entry["rewrite_prompt"] in df.index
                        else None,
                    }
                except KeyError:
                    continue

                if not all(
                    isinstance(value, str) for value in extracted_entry.values()
                ):
                    continue
                entries.append(extracted_entry)

            except json.JSONDecodeError:
                continue

# %%
# Create a HuggingFace Dataset from the extracted entries
dataset = Dataset.from_list(entries)

# %%
from datasets import DatasetDict
import random

# Set the random seed for reproducibility
random.seed(42)

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2)

# Create a DatasetDict from the split datasets
dataset_dict = DatasetDict(
    {"train": train_test_split["train"], "test": train_test_split["test"]}
)

# Save the dataset to a file
dataset_dict.save_to_disk(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/gemini_categorized_prompts/dataset_train_test_split.hf"
)

# %%
from datasets import load_dataset

# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub(
    "mysfi/curious-bohr", token="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"
)


# %%
# Print 10 random samples from the dataset
print("Random samples from the dataset:")
for sample in dataset.shuffle(seed=42).select(range(10)):
    print(f"Rewrite Prompt: {sample['rewrite_prompt']}")
    print("*" * 10)
    print(f"Original Text: {sample['original_text']}")
    print("*" * 10)
    print(f"Rewritten Text: {sample['rewritten_text']}")
    print("*" * 10)
    print(f"Category: {sample['category']}")
    print("---")

# %%
