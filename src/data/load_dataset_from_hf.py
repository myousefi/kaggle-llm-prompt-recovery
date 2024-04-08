# %%
from random import shuffle
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load the dataset in streaming mode
dataset = load_dataset("allenai/c4", "en", split='train', streaming=True, ).shuffle(buffer_size=1_000, seed=42)

# Since the dataset is in streaming mode, we don't need a DataLoader to iterate over it
i = 0
for batch in dataset:
    if len(batch["text"]) < 400:
        print(batch)

    i += 1
    if i > 100:
        break

# %%

base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
next(iter(pile_dataset["train"]))