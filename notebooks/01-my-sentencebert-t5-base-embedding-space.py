# %%
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file
from sentence_transformers import SentenceTransformer

csv_file_path = "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/interim/gemini_categorized_prompts/prompts.csv"

df = pd.read_csv(csv_file_path, delimiter=";")

# Load the Sentence T5 model
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
# Perform t-SNE
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeddings)


# Apply cosine^3 transformation to the similarity matrix
cosine_sim_matrix = 1 - cosine_sim_matrix**3

cosine_sim_matrix = np.clip(cosine_sim_matrix, 0, 1)

tsne = TSNE(
    n_components=3,
    verbose=1,
    perplexity=50,
    n_iter=1000,
    metric="precomputed",
    init="random",
)

tsne_results = tsne.fit_transform(cosine_sim_matrix)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(
    tsne_results, columns=["tsne-3d-one", "tsne-3d-two", "tsne-3d-three"]
)
tsne_df["Prompt"] = df["Prompt"]
tsne_df["Property"] = df["Category"]

# %%
# tsne_df.to_csv(
#     "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/processed/t5-embedding-tsne/tsne_results.csv",
#     index=False,
# )
# np.savetxt(
#     "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/processed/t5-embedding-tsne/cosine_sim_matrix.csv",
#     cosine_sim_matrix,
#     delimiter=",",
# )

# %%
# Perform t-SNE
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(embeddings)


# Apply cosine^3 transformation to the similarity matrix
cosine_sim_matrix = 1 - cosine_sim_matrix**3

cosine_sim_matrix = np.clip(cosine_sim_matrix, 0, 1)


tsne = TSNE(
    n_components=3,
    verbose=1,
    perplexity=50,
    n_iter=1000,
    metric="precomputed",
    init="random",
)

tsne_results = tsne.fit_transform(cosine_sim_matrix)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(
    tsne_results, columns=["tsne-3d-one", "tsne-3d-two", "tsne-3d-three"]
)
tsne_df["Prompt"] = df["Prompt"]
tsne_df["Property"] = df["Category"]

# Save the t-SNE results to a CSV file
tsne_df.to_csv("tsne_results.csv", index=False)

# %%
# Visualize with Plotly

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    [
        dcc.Graph(
            id="tsne-3d-plot",
            figure={
                "data": [
                    go.Scatter3d(
                        x=tsne_df["tsne-3d-one"],
                        y=tsne_df["tsne-3d-two"],
                        z=tsne_df["tsne-3d-three"],
                        mode="markers",
                        marker=dict(size=5, opacity=0.5),
                        text=tsne_df["Prompt"],
                        hoverinfo="text",
                    )
                ],
                "layout": go.Layout(
                    margin={"l": 0, "r": 0, "b": 0, "t": 0}, hovermode="closest"
                ),
            },
            style={"height": 600},
            config={"displayModeBar": False},
        ),
        dash_table.DataTable(
            id="similarity-table",
            columns=[
                {"name": "Point Index 1", "id": "point1"},
                {"name": "Point Index 2", "id": "point2"},
                {"name": "Cosine Similarity", "id": "similarity"},
            ],
            data=[],
        ),
    ]
)


# Define callback to update table
@app.callback(
    Output("similarity-table", "data"),
    [Input("tsne-3d-plot", "selectedData")],
    [State("similarity-table", "data")],
)
def display_selected_data(selectedData, rows):
    if selectedData:
        points = selectedData["points"]
        point_indices = [point["pointIndex"] for point in points]
        new_rows = rows.copy()
        for i in range(len(point_indices)):
            for j in range(i + 1, len(point_indices)):
                idx1, idx2 = point_indices[i], point_indices[j]
                similarity = cosine_sim_matrix[idx1, idx2]
                new_rows.append(
                    {"point1": idx1, "point2": idx2, "similarity": similarity}
                )
        return new_rows
    return rows


# Run the app
# if __name__ == "__main__":
import socket


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


app.run_server(debug=True, host="0.0.0.0", port=find_free_port())


# Optionally, save the figure to a file
# fig.write_html('/path/to/tsne_visualization.html')

# %%
# %%
from sklearn_extra.cluster import KMedoids

# Fit KMedoids with the optimal number of clusters
kmedoids = KMedoids(
    n_clusters=20,
    random_state=42,
    metric="precomputed",
)

kmedoids.fit(cosine_sim_matrix)

# Add cluster labels to the DataFrame
tsne_df["Cluster"] = str(kmedoids.labels_)

# Visualize with Plotly
fig = px.scatter_3d(
    tsne_df,
    x="tsne-3d-one",
    y="tsne-3d-two",
    z="tsne-3d-three",
    color="Cluster",
    hover_data=["Prompt", "Property"],
    color_discrete_sequence=px.colors.qualitative.Plotly,
)
fig.show(renderer="browser")

# %%
import plotly.express as px

# Iterate over each cluster
for cluster in tsne_df["Cluster"].unique():
    # Filter the DataFrame for the current cluster
    cluster_df = tsne_df[tsne_df["Cluster"] == cluster]

    # Create a histogram of the Property for the current cluster
    fig = px.histogram(
        cluster_df, x="Property", title=f"Histogram of Property for Cluster {cluster}"
    )
    fig.update_layout(xaxis_title="Property", yaxis_title="Frequency")
    fig.show()
