# %%

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/raw/chatgpt_generated_prompts_.csv', delimiter=";")

df.head()


# %%
from transformers import T5Tokenizer, T5Model
import torch

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5Model.from_pretrained('t5-base')

# Function to calculate embeddings
def calculate_t5_embeddings(texts):
    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    # Get the embeddings from the last hidden state
    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=inputs["input_ids"])
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# %%
import time
import matplotlib.pyplot as plt

# Sample a subset of the DataFrame for timing
sample_df = df.sample(n=100)

# Initialize a list to store timings
timings = []

# Calculate embeddings for the "Prompt" column in the sample DataFrame and time the process
for prompt in sample_df['Prompt']:
    start_time = time.time()
    embedding = calculate_t5_embeddings(prompt).numpy()
    end_time = time.time()
    timings.append(end_time - start_time)

# Add the embeddings as a new column in the sample DataFrame
sample_df['t5_base_embeddings'] = sample_df['Prompt'].apply(lambda x: calculate_t5_embeddings([x]).numpy())

# Plot the timings
plt.plot(timings)
plt.xlabel('Sample Index')
plt.ylabel('Time to Calculate Embedding (seconds)')
plt.title('Time to Calculate T5 Embeddings for Sample Prompts')
plt.show()

# %%
# Add the embeddings as a new column in the original DataFrame
sample_df.iloc[0,2].shape
# %%
