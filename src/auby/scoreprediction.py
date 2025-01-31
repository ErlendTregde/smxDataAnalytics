import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load JSON Data
DATA_DIR = "./data"
FILE = "auby_latest_200.json"
file_path = os.path.join(DATA_DIR, FILE)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.json_normalize(data)

# Select Features
features = ["chart.difficulty", "max_combo", "perfect1", "perfect2", "misses", "early", "late", "chart.play_count"]
X = df[features]
y = df["score"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances, color="teal")
plt.xlabel("Importance Score")
plt.title("Feature Importance for Score Prediction - Auby")
plt.show()
