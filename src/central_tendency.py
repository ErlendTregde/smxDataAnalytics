import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to process data
def flatten_scores(data):
    df = pd.json_normalize(data)
    return df[["gamer.username", "score", "chart.difficulty"]] if "score" in df.columns else df

# Load and process data
dfs = {key: flatten_scores(load_json(os.path.join(DATA_DIR, filename))) for key, filename in FILES.items()}

# Plot violin plots to show score distribution (mean vs median)
plt.figure(figsize=(12, 6))
for key, df in dfs.items():
    sns.violinplot(x=[key] * len(df), y=df["score"], inner="quartile")
plt.title("Score Distribution (Mean vs Median)")
plt.xlabel("Dataset")
plt.ylabel("Score")
plt.show()
