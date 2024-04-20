# %%
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# Load the t-SNE results and cosine similarity matrix
tsne_df = pd.read_csv(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/processed/t5-embedding-tsne/tsne_results.csv"
)

# Define a dictionary mapping each prompt to its index
prompt_to_index = {prompt: index for index, prompt in enumerate(tsne_df["Prompt"])}

cosine_sim_matrix = np.loadtxt(
    "/scratch/yousefi.m/projects/kaggle_llm_prompt_recovery/data/processed/t5-embedding-tsne/cosine_sim_matrix.csv",
    delimiter=",",
)

# Initialize the Dash app
app = dash.Dash(__name__)

import matplotlib.pyplot as plt
import matplotlib

# Generate a colormap with as many colors as there are unique categories
num_categories = len(tsne_df["Property"].unique())
cmap = plt.cm.get_cmap(
    "gist_rainbow", num_categories
)  # Using 'gist_rainbow' for extremely distinct colors

# Create a color map from the colormap
color_map = {
    category: matplotlib.colors.rgb2hex(cmap(i))
    for i, category in enumerate(tsne_df["Property"].unique())
}

# Define the layout of the app
app.layout = html.Div(
    [
        dcc.Graph(
            id="tsne-3d-plot",
            figure={
                "data": [
                    go.Scatter3d(
                        x=tsne_df[tsne_df["Property"] == category]["tsne-3d-one"],
                        y=tsne_df[tsne_df["Property"] == category]["tsne-3d-two"],
                        z=tsne_df[tsne_df["Property"] == category]["tsne-3d-three"],
                        mode="markers",
                        marker=dict(
                            size=5,
                            opacity=0.5,
                            color=color_map.get(
                                category, "#000000"
                            ),  # Assign color from color_map, default to black if not existent
                        ),
                        name=category,  # Name of the trace is the category
                        text=tsne_df[tsne_df["Property"] == category]["Prompt"],
                        hoverinfo="text",
                    )
                    for category in tsne_df[
                        "Property"
                    ].unique()  # Create a trace for each unique category
                ],
                "layout": go.Layout(
                    clickmode="event+select",
                    margin={"l": 0, "r": 0, "b": 0, "t": 0},
                    hovermode="closest",
                    legend=dict(
                        title="Property",  # Optional: add a title to the legend
                        itemsizing="constant",  # Optional: ensure consistent size for legend items
                    ),
                ),
            },
            style={"height": 600},
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                "doubleClick": "reset",
            },
        ),
        dash_table.DataTable(
            id="similarity-table",
            columns=[
                {"name": "Point Index 1", "id": "point1"},
                {"name": "Prompt 1", "id": "prompt1"},
                {"name": "Point Index 2", "id": "point2"},
                {"name": "Prompt 2", "id": "prompt2"},
                {"name": "Cosine Similarity", "id": "similarity"},
            ],
            data=[],
            style_table={"overflowX": "auto"},
            style_cell={
                "height": "auto",
                "minWidth": "180px",
                "width": "180px",
                "maxWidth": "180px",
                "whiteSpace": "normal",
            },
        ),
    ]
)

from collections import deque

# Initialize a deque with a maximum length of 2 to store the last two clicked points
last_two_clicks = deque(maxlen=2)


@app.callback(
    [Output("similarity-table", "data")],
    [Input("tsne-3d-plot", "clickData")],
    [State("tsne-3d-plot", "figure"), State("similarity-table", "data")],
)
def display_clicked_data(clickData, figure, rows):
    if clickData:
        # Get the index of the clicked point and add it to the deque
        # idx = clickData["points"][0]["pointIndex"]
        last_two_clicks.append(clickData)

        # If we have two points, calculate the similarity
        if len(last_two_clicks) == 2:
            prompt1, prompt2 = (
                last_two_clicks[0]["points"][0]["text"],
                last_two_clicks[1]["points"][0]["text"],
            )

            idx1 = prompt_to_index[prompt1]
            idx2 = prompt_to_index[prompt2]

            similarity = 1 - cosine_sim_matrix[idx1, idx2]

            table_data = [
                {
                    "point1": idx1,
                    "prompt1": prompt1,
                    "point2": idx2,
                    "prompt2": prompt2,
                    "similarity": similarity,
                }
            ]
        else:
            table_data = rows  # Keep the existing data if we don't have two points yet
    else:
        table_data = rows  # Use the existing data if there is no clickData

    return [table_data]  # Always return a list with a single element


import socket


def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


# %%
if __name__ == "__main__":
    port = find_free_port()
    app.run_server(debug=False, port=port)
