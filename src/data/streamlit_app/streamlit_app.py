import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Streamlit UI setup
st.set_page_config(page_title="LLM Prompt Recovery Playground")
st.title("LLM Prompt Recovery Playground")


# Function to load model and tokenizer, cached with st.cache
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config
    )
    return tokenizer, model


model_id = "google/gemma-7b-it"
tokenizer, model = load_model_and_tokenizer(model_id)

# Load prompts DataFrame
prompts_df = pd.read_csv(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/raw/gemini_generated_prompts.csv",
    delimiter=";",
)

# Rewrite Prompt Selection UI
st.subheader("Rewrite Prompt Selection")
use_custom_prompt = st.checkbox("Write your own rewrite prompt")
if use_custom_prompt:
    rewrite_prompt = st.text_area(
        "Write your rewrite prompt:", "Type your prompt here..."
    )
else:
    rewrite_prompt = st.selectbox("Choose a rewrite prompt:", prompts_df["Prompt"])

# User input for the original text and display the rewritten text side-by-side
st.subheader("Original and Rewritten Text")
col1, col2 = st.columns(2)
with col1:
    user_input = st.text_area("Enter the original text:", "Type your text here...")
with col2:
    if st.button("Rewrite Text"):
        # Create the prompt and generate output
        prompt = f"""<start_of_turn>user
{rewrite_prompt}:
{user_input}<end_of_turn>
<start_of_turn>model
        """
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(
            **inputs,
            max_new_tokens=len(inputs),
            do_sample=False,
        )
        rewritten_text = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        )

        st.text_area("Rewritten Text:", rewritten_text, height=300)

# Note on running the Streamlit app with a specific server address:
# To run this Streamlit app on address 0.0.0.0, use the command:
# streamlit run streamlit_app.py --server.address 0.0.0.0
# This command must be executed in the terminal, not in the script.
