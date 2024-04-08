# %%
import pandas as pd

df = pd.read_csv(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/raw/chatgpt_generated_prompts_.csv",
    delimiter=";",
)

# %%
import pandas as pd
import torch
from tqdm import tqdm

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/sentence-t5-base")

# Calculate embeddings with batch processing
batch_size = 64  # You can adjust the batch size according to your system's capabilities
embeddings = []
for i in tqdm(range(0, len(df["Prompt"]), batch_size)):
    batch = df["Prompt"][i : i + batch_size].tolist()
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    embeddings.extend(batch_embeddings)

df["embeddings"] = list(embeddings)

# %%
import plotly.express as px
import umap
import numpy as np
# from umap import UMAP

# Perform UMAP dimensionality reduction
umap_embeddings = umap.UMAP(
    n_components=3,
    metric=lambda x, y: (1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    ** 3,
).fit_transform(df["embeddings"].tolist())

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(
    x=umap_embeddings[:, 0],
    y=umap_embeddings[:, 1],
    z=umap_embeddings[:, 2],
    color=df["Property"],
    hover_name=df["Prompt"],
    title="UMAP Visualization of Embeddings",
)

# Update the plot layout
fig.update_layout(
    scene=dict(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        zaxis_title="UMAP Dimension 3",
    ),
    width=800,
    height=600,
)

# Display the plot
fig.show(renderer="browser")

# %%
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import numpy as np

# Perform HDBSCAN clustering
clusterer = HDBSCAN(
    min_cluster_size=5,
    metric=lambda x, y: (1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    ** 3,
)

clusterer.fit(df["embeddings"].tolist())

# Add the cluster labels to the DataFrame
df["cluster"] = clusterer.labels_

# Print the cluster assignments
print(df[["rewrite_prompt", "cluster"]])

# Plot the number of clusters and cluster sizes
unique_labels = set(clusterer.labels_)
cluster_sizes = [sum(clusterer.labels_ == label) for label in unique_labels]

plt.figure(figsize=(8, 6))
plt.bar(range(len(unique_labels)), cluster_sizes)
plt.xlabel("Cluster")
plt.ylabel("Size")
plt.title("Cluster Sizes")
plt.xticks(range(len(unique_labels)), unique_labels)
plt.show()
# %%
