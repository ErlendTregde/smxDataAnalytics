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
    columns_to_keep = ["score"]
    return df[columns_to_keep] if set(columns_to_keep).issubset(df.columns) else df

# Load data
dfs = {key: flatten_scores(load_json(os.path.join(DATA_DIR, filename))) for key, filename in FILES.items()}

# Detect outliers using boxplots
plt.figure(figsize=(10,6))
sns.boxplot(data=[df["score"] for df in dfs.values()], palette="Set2")
plt.xticks(ticks=[0,1,2], labels=dfs.keys())
plt.title("Outlier Detection: Score Distribution")
plt.ylabel("Score")
plt.show()
