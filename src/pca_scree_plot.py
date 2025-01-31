import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths to JSON files
DATA_DIR = "./data"
FILE = "100ScoresAuby.json"  # Using Auby's dataset for PCA

# Function to load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load and preprocess data
file_path = os.path.join(DATA_DIR, FILE)
df = pd.json_normalize(load_json(file_path))

# Selecting numerical features for PCA
features = ["score", "max_combo", "perfect1", "perfect2", "misses"]
df_pca = df[features]

# Standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)

# Performing PCA
pca = PCA()
pca.fit(df_scaled)

# Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(features) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot - Variance Explained by Principal Components")
plt.xticks(range(1, len(features) + 1))
plt.grid(True)
plt.show()

# Print variance explained by each component
print("\nPCA Explained Variance Ratio:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"Principal Component {i+1}: {variance:.4f} ({variance*100:.2f}%)")
