# %% [code] {"papermill":{"duration":0.012141,"end_time":"2024-03-01T09:43:07.781271","exception":false,"start_time":"2024-03-01T09:43:07.76913","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:19:24.621205Z","iopub.execute_input":"2024-03-06T10:19:24.621906Z","iopub.status.idle":"2024-03-06T10:19:24.625893Z","shell.execute_reply.started":"2024-03-06T10:19:24.621875Z","shell.execute_reply":"2024-03-06T10:19:24.624876Z"}}
### CREDITS
### https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
### https://www.kaggle.com/code/aatiffraz/prompt-prediction-w-mixtral-mistral7b-gemma-llama/notebook

# %% [code] {"papermill":{"duration":54.916984,"end_time":"2024-03-01T09:44:02.702691","exception":false,"start_time":"2024-03-01T09:43:07.785707","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:19:24.627731Z","iopub.execute_input":"2024-03-06T10:19:24.628131Z","iopub.status.idle":"2024-03-06T10:20:16.247004Z","shell.execute_reply.started":"2024-03-06T10:19:24.628084Z","shell.execute_reply":"2024-03-06T10:20:16.245949Z"}}

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true,"papermill":{"duration":6.461738,"end_time":"2024-03-01T09:44:09.16901","exception":false,"start_time":"2024-03-01T09:44:02.707272","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:20:16.248961Z","iopub.execute_input":"2024-03-06T10:20:16.249261Z","iopub.status.idle":"2024-03-06T10:20:16.253987Z","shell.execute_reply.started":"2024-03-06T10:20:16.249234Z","shell.execute_reply":"2024-03-06T10:20:16.252896Z"}}
import bitsandbytes
import accelerate
import transformers
import optimum

# %% [code] {"papermill":{"duration":109.931476,"end_time":"2024-03-01T09:45:59.104992","exception":false,"start_time":"2024-03-01T09:44:09.173516","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:20:16.255229Z","iopub.execute_input":"2024-03-06T10:20:16.255551Z","iopub.status.idle":"2024-03-06T10:22:15.181408Z","shell.execute_reply.started":"2024-03-06T10:20:16.255520Z","shell.execute_reply":"2024-03-06T10:22:15.180587Z"}}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig


# Comment/Uncomment and use as per wish

MODEL_PATH = "google/gemma-7b-it"
# MODEL_PATH = "/kaggle/input/gemma/transformers/7b-it/1"
# # MODEL_PATH = "/kaggle/input/gemma/transformers/2b-it/2"
# MODEL_PATH = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"
# MODEL_PATH = "/kaggle/input/mixtral/pytorch/8x7b-instruct-v0.1-hf/1"
# MODEL_PATH = "/kaggle/input/phi/transformers/2/1"

# Found a good blog to catch me up fast!
# https://huggingface.co/blog/4bit-transformers-bitsandbytes
# https://huggingface.co/docs/transformers/v4.38.1/en/quantization#compute-data-type
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=3072)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map = "auto",
    trust_remote_code = True,
    quantization_config=quantization_config,
)

# model = model.to_bettertransformer()

# %% [code] {"papermill":{"duration":12.816958,"end_time":"2024-03-01T09:46:11.926766","exception":false,"start_time":"2024-03-01T09:45:59.109808","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:15.183982Z","iopub.execute_input":"2024-03-06T10:22:15.184404Z","iopub.status.idle":"2024-03-06T10:22:15.201042Z","shell.execute_reply.started":"2024-03-06T10:22:15.184371Z","shell.execute_reply":"2024-03-06T10:22:15.200138Z"}}
import pandas as pd
from string import Template
from pathlib import Path
import numpy as np
import os

import warnings
warnings.simplefilter("ignore")

import torch
from transformers import pipeline, AutoTokenizer

from tqdm.notebook import tqdm

data_path = Path('data/external/kaggle/input/llm-prompt-recovery')

test = pd.read_csv(data_path / 'test.csv', index_col='id')
test.head()

# %% [code] {"papermill":{"duration":0.015272,"end_time":"2024-03-01T09:46:11.947006","exception":false,"start_time":"2024-03-01T09:46:11.931734","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:15.202238Z","iopub.execute_input":"2024-03-06T10:22:15.202503Z","iopub.status.idle":"2024-03-06T10:22:15.209981Z","shell.execute_reply.started":"2024-03-06T10:22:15.202481Z","shell.execute_reply":"2024-03-06T10:22:15.209139Z"}}
from torch import nn
class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        #perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity 
    
perp = Perplexity()

# %% [code] {"papermill":{"duration":0.011681,"end_time":"2024-03-01T09:46:11.963396","exception":false,"start_time":"2024-03-01T09:46:11.951715","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:15.211273Z","iopub.execute_input":"2024-03-06T10:22:15.211890Z","iopub.status.idle":"2024-03-06T10:22:15.224388Z","shell.execute_reply.started":"2024-03-06T10:22:15.211858Z","shell.execute_reply":"2024-03-06T10:22:15.223629Z"}}
def format_prompt(row, prompt):
    prompt_ = f"""<start_of_turn>user
{prompt}
{row["original_text"]}<end_of_turn>
<start_of_turn>model
{row["rewritten_text"]}<end_of_turn>"""
    return prompt_

def format_the_rewriting(original_text, prompt):
    prompt_ = f"""<start_of_turn>user
                {prompt}:
                {original_text}<end_of_turn>
                <start_of_turn>model"""
    return prompt_

# %% [code] {"papermill":{"duration":0.011349,"end_time":"2024-03-01T09:46:11.979425","exception":false,"start_time":"2024-03-01T09:46:11.968076","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:15.225402Z","iopub.execute_input":"2024-03-06T10:22:15.225656Z","iopub.status.idle":"2024-03-06T10:22:15.234776Z","shell.execute_reply.started":"2024-03-06T10:22:15.225635Z","shell.execute_reply":"2024-03-06T10:22:15.234034Z"}}
rewrite_prompts = [
    # 'Convert this to a sea shanty',
    # 'Please improve the following text using the writing style of, maintaining the original meaning but altering the tone, diction, and stylistic elements to match the new style.Enhance the clarity, elegance, and impact of the following text by adopting the writing style of , ensuring the core message remains intact while transforming the tone, word choice, and stylistic features to align with the specified style.',
    "Transform this into a psychological drama, delving into the depths of the human psyche and the complexities of relationships and identity.", 
    "Chisel the text to radiate with sophistication.",
    # "Engrave the text with a hint of mischief.",
    # "Sketch the text to whisper with nostalgia.",
    "Embroider the text with a sense of wonder.",
    "Garnish the text with a sprinkle of magic.",
]

# %%
for idx, row in tqdm(test.iterrows(), total=len(test)):
        
    
    with torch.no_grad():
        perps = []
        samples = []
        for prompt in rewrite_prompts:
            formatted_prompt = format_the_rewriting(row["original_text"], prompt)
            # break

            inputs = tokenizer(formatted_prompt, return_tensors="pt").to('cuda')

            outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

            rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

            print(rewritten_text)

    break
# %% [code] {"papermill":{"duration":3.907872,"end_time":"2024-03-01T09:46:15.892143","exception":false,"start_time":"2024-03-01T09:46:11.984271","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:15.235782Z","iopub.execute_input":"2024-03-06T10:22:15.236034Z","iopub.status.idle":"2024-03-06T10:22:19.470610Z","shell.execute_reply.started":"2024-03-06T10:22:15.236013Z","shell.execute_reply":"2024-03-06T10:22:19.469591Z"}}
preds = []

for idx, row in tqdm(test.iterrows(), total=len(test)):
        
    
    with torch.no_grad():
        perps = []
        samples = []
        for prompt in rewrite_prompts:
            samples.append(format_prompt(row, prompt))
        inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to("cuda")

        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        output = output.logits
        labels = inputs["input_ids"]
        labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
        for j in range(len(rewrite_prompts)):
            p = perp(output[j].unsqueeze(0), labels[j].unsqueeze(0))
            perps.append(p.detach().cpu())
            
        # del inputs
        # del labels
        # del output
        # del p

    perps = np.array(perps)
        
    predictions = [np.array(rewrite_prompts)[np.argsort(perps)][0]]
    preds.append(predictions[0])
    print(preds)

# %% [code] {"papermill":{"duration":0.01536,"end_time":"2024-03-01T09:46:15.912885","exception":false,"start_time":"2024-03-01T09:46:15.897525","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:19.471825Z","iopub.execute_input":"2024-03-06T10:22:19.472134Z","iopub.status.idle":"2024-03-06T10:22:19.479558Z","shell.execute_reply.started":"2024-03-06T10:22:19.472085Z","shell.execute_reply":"2024-03-06T10:22:19.478537Z"}}
submission = pd.read_csv(data_path / 'sample_submission.csv')
submission["rewrite_prompt"] = preds

# %% [code] {"papermill":{"duration":0.015521,"end_time":"2024-03-01T09:46:15.933505","exception":false,"start_time":"2024-03-01T09:46:15.917984","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:19.482012Z","iopub.execute_input":"2024-03-06T10:22:19.482313Z","iopub.status.idle":"2024-03-06T10:22:19.492588Z","shell.execute_reply.started":"2024-03-06T10:22:19.482289Z","shell.execute_reply":"2024-03-06T10:22:19.491682Z"}}
submission.head()

# %% [code] {"papermill":{"duration":0.014739,"end_time":"2024-03-01T09:46:15.953356","exception":false,"start_time":"2024-03-01T09:46:15.938617","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2024-03-06T10:22:19.493901Z","iopub.execute_input":"2024-03-06T10:22:19.494174Z","iopub.status.idle":"2024-03-06T10:22:19.502996Z","shell.execute_reply.started":"2024-03-06T10:22:19.494152Z","shell.execute_reply":"2024-03-06T10:22:19.502200Z"}}
submission.to_csv('submission.csv', index=False)