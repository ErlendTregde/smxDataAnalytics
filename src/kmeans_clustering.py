import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Paths to JSON files
DATA_DIR = "./data"
FILE = "100ScoresAuby.json"

# Load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load and preprocess data
file_path = os.path.join(DATA_DIR, FILE)
df = pd.json_normalize(load_json(file_path))

# Selecting numerical features for clustering
features = ["score", "max_combo", "perfect1", "perfect2", "misses"]
df_pca = df[features]

# Standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)

# Applying PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
df_pca_result = pd.DataFrame(principal_components, columns=["PC1", "PC2"])

# Finding optimal clusters using Elbow Method
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(df_pca_result)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), wcss, marker='o', linestyle='-')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method - Finding Optimal K")
plt.grid(True)
plt.show()
