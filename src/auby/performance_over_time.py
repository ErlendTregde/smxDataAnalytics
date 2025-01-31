import json
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_FILE = "./data/auby_latest_200.json"

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

df = pd.json_normalize(load_json(DATA_FILE))

df["created_at"] = pd.to_datetime(df["created_at"])
df = df.sort_values("created_at")

plt.figure(figsize=(12, 6))
plt.plot(df["created_at"], df["score"], marker="o", linestyle="-", alpha=0.75)
plt.xlabel("Time")
plt.ylabel("Score")
plt.title("Auby's Performance Over Time")
plt.xticks(rotation=45)
plt.show()
