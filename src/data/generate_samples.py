# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

model_id = "google/gemma-7b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

# %%
# %%
from random import shuffle
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

# Load the dataset in streaming mode
dataset = load_dataset("allenai/c4", "en", split='train', streaming=True)
dataset = dataset.filter(lambda entry: len(entry["text"]) < 1_000)

dataset = dataset.shuffle(buffer_size=10_000, seed=int(os.environ.get('SLURM_JOB_ID')))

# %%
import pandas as pd

# Read the CSV file into a DataFrame
gemini_generated_prompts = pd.read_csv('/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/raw/chatgpt_generated_prompts_.csv', delimiter=';')

# %%

import json

device = "cuda:0"
output_file_path = f'/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/chatgpt/gemma_7b_it_rewrites_{os.environ.get("SLURM_JOB_ID")}.json'

# %%
while True:
    
    for original_text in iter(dataset):
        rewrite_prompt = gemini_generated_prompts.sample(1)["Prompt"].values[0]
        prompt = f"""<start_of_turn>user
{rewrite_prompt}:
{original_text}<end_of_turn>
<start_of_turn>model
        """

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=2 * len(original_text['text']), do_sample=False)

        rewritten_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        data_entry = {
            'rewrite_prompt': rewrite_prompt,
            "original_text": original_text['text'],
            "rewritten_text": rewritten_text
        }


        with open(output_file_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
# %%