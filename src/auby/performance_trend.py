import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import linregress
from sklearn.cluster import KMeans

# Load data
DATA_DIR = "./data"
FILE = "auby_latest_200.json"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_json(os.path.join(DATA_DIR, FILE))
df = pd.json_normalize(data)

# Convert timestamp to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at')

# Performance Over Time with Trend Line
slope, intercept, _, _, _ = linregress(range(len(df)), df['score'])
df['trend'] = slope * range(len(df)) + intercept
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['created_at'], y=df['score'], label='Actual Scores', marker='o')
sns.lineplot(x=df['created_at'], y=df['trend'], label='Trend Line', linestyle='dashed')
plt.xticks(rotation=45)
plt.title("Performance Over Time - Auby")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.show()

# Rolling Average of Scores
df['rolling_avg'] = df['score'].rolling(window=10).mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['created_at'], y=df['score'], label='Actual Scores', marker='o')
sns.lineplot(x=df['created_at'], y=df['rolling_avg'], label='Rolling Average (10 games)', linestyle='dashed')
plt.xticks(rotation=45)
plt.title("Rolling Average of Scores - Auby")
plt.xlabel("Date")
plt.ylabel("Score")
plt.legend()
plt.show()

# Clustering Scores by Attributes
features = ["score", "max_combo", "perfect1", "perfect2", "misses", "chart.difficulty"]
df_clustering = df[features]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['score'], y=df['chart.difficulty'], hue=df['cluster'], palette='viridis')
plt.title("Clustering Players by Score and Difficulty")
plt.xlabel("Score")
plt.ylabel("Difficulty")
plt.legend(title="Cluster")
plt.show()

# Feature Importance for Score Prediction
features = ["max_combo", "perfect1", "perfect2", "misses", "chart.difficulty", "chart.play_count"]
target = "score"
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'], palette='magma')
plt.title("Feature Importance for Score Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Outlier Detection in Performance
sns.boxplot(x=df['score'])
plt.title("Outlier Detection in Scores - Auby")
plt.xlabel("Score")
plt.show()
