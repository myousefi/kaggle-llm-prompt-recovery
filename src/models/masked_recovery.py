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
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")

# %%
# Example input
input_text = '''Make this text intentionally melodramatic and over-the-top:"""Jodi joined Zimmer Insurance Group in 2006 providing support and assistance in the Commercial Lines Department as a Customer Service Rep. Jodi obtained her Producer’s license in 1994 and has 18 years of experience in the Insurance Industry.\nIn addition to working with many of our long time clients, Jodi is eager to develop new relationships as we continue to grow.\nJodi and her husband enjoy country living and they enjoy many outdoor activities and spending time with their animals."""\n\nSure, here is the text with the added melodramatic flair:\n\n\"Jodi joined Zimmer Insurance Group in 2006, a siren song of support and assistance in the Commercial Lines Department as a Customer Service Rep. With a voice like honeyed honey, she provided a symphony of service, a melody of compassion, and a harmony of understanding.\n\nIn 1994, she obtained her Producer’s license, a testament to her unwavering determination and a testament to her 18 years of experience in the Insurance Industry. With a resume as impressive as a diamond necklace, she has a track record of success that would make even the most jaded cynic believe in the power of a smile.\n\nBeyond her professional accomplishments, Jodi is a woman of boundless energy and boundless compassion. She is a woman who can tame the wildest of beasts, a woman who can bring joy to even the grumpiest of souls. And when she's not busy saving the world, she enjoys country living, outdoor activities, and spending time with her furry companions.\n\nSo, if you're looking for an insurance agent who can provide you with the best possible service, look no further than Jodi. She's the one who can make your dreams a reality, one melodramatic sentence at a time.\""'''

# Tokenize the input
input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

# Create an attention mask
attention_mask = tokenizer(input_text, return_tensors="pt")["attention_mask"]

# Identify the word to mask in the input text
word_to_mask = "intentionally"  # Replace with the word you want to mask
word_index = tokenizer(input_text, return_tensors="pt").input_ids[0].tolist().index(tokenizer.encode(word_to_mask, add_special_tokens=False)[0])

# Mask the identified word
attention_mask[:, word_index] = 0

# Pass the input and attention mask to the model
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Extract the logits for the masked word
masked_word_logits = outputs.logits[:, word_index]

# Optionally, decode the logits to get the predicted token
predicted_word = tokenizer.decode(masked_word_logits.argmax(-1))

print(predicted_word)


# %%
# Prepare the input text
input_text = "The quick brown fox jumps over the <unk> dog."

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get the masked token position
# Get the masked token position
masked_position = (input_ids == tokenizer.unk_token_id).nonzero()[0][-1].item()

# Pass the input to the model and get the output logits
with torch.no_grad():
    output = model(input_ids)
    logits = output.logits

# Extract the logits for the masked position
masked_logits = logits[0, masked_position]

# Convert logits to probabilities
probabilities = torch.softmax(masked_logits, dim=-1)

# Get the top predictions
top_predictions = torch.topk(probabilities, k=5)

# Print the top predictions
for idx, prob in zip(top_predictions.indices, top_predictions.values):
    token = tokenizer.decode([idx])
    print(f"{token}: {prob.item():.4f}")