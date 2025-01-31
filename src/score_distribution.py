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
    return df[["score"]] if "score" in df.columns else df

# Load data
dfs = {key: flatten_scores(load_json(os.path.join(DATA_DIR, filename))) for key, filename in FILES.items()}

# Plot score distributions
plt.figure(figsize=(12,6))
for key, df in dfs.items():
    sns.histplot(df["score"], bins=20, kde=True, label=key)
plt.title("Score Distribution Across Different Datasets")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()
