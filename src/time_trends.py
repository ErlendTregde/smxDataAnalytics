import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Paths to JSON files
DATA_DIR = "./data"
FILE = "100ScoresAuby.json"  # Focusing on Auby's data

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load data
file_path = os.path.join(DATA_DIR, FILE)
df = pd.json_normalize(load_json(file_path))

# Convert timestamps
df["created_at"] = pd.to_datetime(df["created_at"])

# Plot score progression over time
plt.figure(figsize=(12, 6))
plt.plot(df["created_at"], df["score"], marker="o", linestyle="-")
plt.title("Player Performance Over Time")
plt.xlabel("Time")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()
