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
    columns_to_keep = ["score", "chart.difficulty"]
    return df[columns_to_keep] if set(columns_to_keep).issubset(df.columns) else df

# Load data
dfs = {key: flatten_scores(load_json(os.path.join(DATA_DIR, filename))) for key, filename in FILES.items()}

# Plot difficulty vs. score
plt.figure(figsize=(12,6))
for key, df in dfs.items():
    sns.scatterplot(x=df["chart.difficulty"], y=df["score"], label=key)
plt.title("Impact of Difficulty on Score")
plt.xlabel("Difficulty")
plt.ylabel("Score")
plt.legend()
plt.show()
