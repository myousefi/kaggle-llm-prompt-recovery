# %%
from datasets import load_from_disk

dataset = load_from_disk(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/gemini_categorized_prompts/dataset_train_test_split.hf",
)

samples = dataset["train"].shuffle(seed=42).select(range(1000))


# %%
from sentence_transformers import SentenceTransformer

# Load the Sentence T5 model
model = SentenceTransformer("sentence-transformers/sentence-t5-base")

# Calculate embeddings for the original and rewritten prompts
original_embeddings = model.encode(samples["original_text"])
rewritten_embeddings = model.encode(samples["rewritten_text"])

# %%
# Define the change direction vector
change_direction = rewritten_embeddings - original_embeddings

# %%
from sklearn.manifold import TSNE
import plotly.express as px
import numpy as np

# Assuming change_direction is a NumPy array of embeddings
# If it's not, you might need to convert it first
change_direction_np = np.array(change_direction)

# Apply t-SNE for dimensionality reduction
# You might need to adjust the parameters based on your data
tsne = TSNE(n_components=2, random_state=42)
change_direction_tsne = tsne.fit_transform(change_direction_np)

# Create a DataFrame for Plotly
import pandas as pd
# Create a DataFrame for Plotly

df = pd.DataFrame(change_direction_tsne, columns=["x", "y"])
df["original_text"] = samples["original_text"]
df["rewritten_text"] = samples["rewritten_text"]
df["rewrite_prompt"] = samples["rewrite_prompt"]
# Assuming you have a column in your DataFrame called 'category' that contains the category for each sample
df["category"] = samples["category"]


# Function to insert <br> tags after every 300 characters or at the end of the text
def insert_line_breaks(text):
    if len(text) <= 300:
        return text
    else:
        lines = []
        for i in range(0, len(text), 300):
            lines.append(text[i : i + 300])
        return "<br>".join(lines)


# Create the hover text
hover_text = []
for index, row in df.iterrows():
    original_text = insert_line_breaks(row["original_text"])
    rewritten_text = insert_line_breaks(row["rewritten_text"])
    rewrite_prompt = insert_line_breaks(row["rewrite_prompt"])

    text = (
        f"<b>Category:</b> {row['category']}<br><br>"
        f"<b>Original Text:</b><br>{original_text}<br><br>"
        f"<b>Rewritten Text:</b><br>{rewritten_text}<br><br>"
        f"<b>Rewrite Prompt:</b><br>{rewrite_prompt}"
    )
    hover_text.append(text)

df["hover_text"] = hover_text

# Visualize with Plotly
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="category",
    title="t-SNE Visualization of Change Direction Embeddings",
    hover_data={
        "hover_text": True,
        "x": False,
        "y": False,
    },
)

# Customize the hover tooltip
fig.update_traces(hovertemplate="%{customdata[0]}")

fig.show(renderer="browser")
