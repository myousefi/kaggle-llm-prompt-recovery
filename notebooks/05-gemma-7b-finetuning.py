# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "google/gemma-2b-it"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
)

# %%
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)


# %%

alpaca_prompt = """You are a prompt engineer. Examine the original and rewritten texts to identify differences in style, such as tone, vocabulary, and structure. Consider the changes made from the original to the rewritten text, and deduce the objectives behind these modifications. Based on this analysis, infer the rewriting prompt that guided the transformation, focusing on the desired outcome like simplification, formalization, or stylistic adaptation. Your task is to articulate the rewriting prompt that logically connects the original text with its rewritten version.

### original text:
{}

### rewritten text:
{}

Inferred from the stylistic and content analysis, the rewriting prompt likely was
### Rewriting Prompt:
{}
"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


def formatting_prompts_func(examples):
    instructions = examples["original_text"]
    inputs = examples["rewritten_text"]
    outputs = examples["rewrite_prompt"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


pass

from datasets import load_from_disk

dataset = load_from_disk(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/gemini_categorized_prompts/dataset_train_test_split.hf",
)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    num_proc=8,
)
# %%
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=300,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        run_name="gemma-2b-it-finetuning",
        save_steps=50,
        save_total_limit=1,
    ),
    peft_config=lora_config,
    # formatting_func=formatting_func,
)

trainer.train()
# %%
import random

test_dataset = dataset["test"]
num_samples = 10


def generate_samples(model, tokenizer, test_dataset, num_samples):
    for i in range(num_samples):
        random_index = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[random_index]

        instruction = sample["original_text"]
        input_text = sample["rewritten_text"]
        rewrite_prompt = sample["rewrite_prompt"]

        prompt = alpaca_prompt.format(instruction, input_text, "")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids, max_length=512)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"Sample {i+1}:")
        print("Instruction:", instruction)
        print("Input:", input_text)
        print("Generated Output:", output)
        print("Rewrite Prompt:", rewrite_prompt)
        print()


generate_samples(model, tokenizer, test_dataset, num_samples)

# %%
