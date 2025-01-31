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

# Running K-Means with optimal K (choosing K=3 from the elbow plot)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_pca_result["Cluster"] = kmeans.fit_predict(df_pca_result)

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster in range(k):
    cluster_data = df_pca_result[df_pca_result["Cluster"] == cluster]
    plt.scatter(cluster_data["PC1"], cluster_data["PC2"], label=f"Cluster {cluster}")

plt.title(f"K-Means Clustering (K={k}) - Player Performance")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
