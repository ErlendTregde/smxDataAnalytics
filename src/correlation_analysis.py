import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to JSON files
DATA_DIR = "./data"
FILES = {
    "auby": "100ScoresAuby.json",
    "same_song": "100scoresSameSongDiffPlayer.json",
    "random": "randomPlayerAndScore.json"
}

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to flatten relevant data fields
def flatten_scores(data):
    df = pd.json_normalize(data)
    columns_to_keep = ["score", "max_combo", "perfect1", "perfect2", "misses"]
    return df[columns_to_keep] if set(columns_to_keep).issubset(df.columns) else df

# Load data
dfs = {key: flatten_scores(load_json(os.path.join(DATA_DIR, filename))) for key, filename in FILES.items()}

# Correlation heatmaps
for key, df in dfs.items():
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap - {key.capitalize()}")
    plt.show()
