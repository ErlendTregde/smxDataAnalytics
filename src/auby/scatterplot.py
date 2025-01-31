import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Load JSON Data
DATA_DIR = "./data"
FILE = "auby_latest_200.json"
file_path = os.path.join(DATA_DIR, FILE)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.json_normalize(data)
df["date"] = pd.to_datetime(df["created_at"])
df = df.sort_values("date")

# Compute Z-Scores for Outlier Detection
df["score_z"] = zscore(df["score"])

# Define threshold (Z > 2 or Z < -2 is considered an outlier)
outliers = df[(df["score_z"] > 2) | (df["score_z"] < -2)]

# Plot Scores with Outliers Highlighted
plt.figure(figsize=(12, 6))
plt.scatter(df["date"], df["score"], label="Scores", alpha=0.6)
plt.scatter(outliers["date"], outliers["score"], color='red', label="Outliers", marker='x')
plt.title("Anomaly Detection in Performance - Auby")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.show()
