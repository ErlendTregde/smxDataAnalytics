import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths to JSON files
DATA_DIR = "./data"
FILE = "100ScoresAuby.json"  # PCA on Auby's dataset

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
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df_pca_result = pd.DataFrame(principal_components, columns=["PC1", "PC2"])

# Print explained variance ratio
print("\n PCA Explained Variance Ratio:")
for i, variance in enumerate(pca.explained_variance_ratio_):
    print(f"Principal Component {i+1}: {variance:.4f} ({variance*100:.2f}%)")

# Plot PCA results
plt.figure(figsize=(10, 6))
plt.scatter(df_pca_result["PC1"], df_pca_result["PC2"])
plt.title("PCA - Player Performance Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
