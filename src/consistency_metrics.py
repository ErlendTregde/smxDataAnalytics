import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to JSON files
DATA_DIR = "./data"
FILE = "100scoresSameSongDiffPlayer.json"  # Analyzing consistency on the same song

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load data
file_path = os.path.join(DATA_DIR, FILE)
df = pd.json_normalize(load_json(file_path))

# Calculate standard deviation of scores per player
player_consistency = df.groupby("gamer.username")["score"].std()

# Visualize consistency per player
plt.figure(figsize=(12, 6))
sns.barplot(x=player_consistency.index, y=player_consistency.values)
plt.xticks(rotation=45)
plt.title("Player Consistency (Standard Deviation of Scores)")
plt.xlabel("Player")
plt.ylabel("Score Standard Deviation")
plt.show()
