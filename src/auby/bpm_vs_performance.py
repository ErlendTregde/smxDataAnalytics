import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = "./data/auby_latest_200.json"

df = pd.json_normalize(json.load(open(DATA_FILE, "r", encoding="utf-8")))

df["song.bpm"] = pd.to_numeric(df["song.bpm"], errors="coerce")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["song.bpm"], y=df["score"])
plt.xlabel("BPM")
plt.ylabel("Score")
plt.title("BPM vs Score")
plt.show()
