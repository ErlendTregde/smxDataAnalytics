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

# Function to flatten relevant data fields
def flatten_scores(data):
    df = pd.json_normalize(data)
    columns_to_keep = [
        "gamer.username", "score", "grade", "max_combo", "full_combo",
        "perfect1", "perfect2", "misses", "early", "late", "song.title",
        "song.bpm", "song.genre", "chart.difficulty", "chart.meter",
        "chart.play_count", "chart.pass_count", "created_at"
    ]
    return df[columns_to_keep] if set(columns_to_keep).issubset(df.columns) else df

# Load and process data
dfs = {}
for key, filename in FILES.items():
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        data = load_json(file_path)
        dfs[key] = flatten_scores(data)
        print(f" Loaded and processed {filename}")
    else:
        print(f" File {filename} not found!")

# Basic statistics
for key, df in dfs.items():
    print(f"\n Summary for {key}:")
    print(df.describe())

# Plot score distributions
plt.figure(figsize=(12,6))
for key, df in dfs.items():
    if "score" in df.columns:
        sns.histplot(df["score"], bins=20, kde=True, label=key)
plt.title("Score Distribution Across Different Datasets")
plt.legend()
plt.show()
