import json
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "./data/auby_latest_200.json"

df = pd.json_normalize(json.load(open(DATA_FILE, "r", encoding="utf-8")))

plt.figure(figsize=(10, 6))
df[["perfect1", "perfect2", "misses", "early", "late"]].mean().plot(kind="bar", color=["green", "blue", "red", "orange", "purple"])
plt.xlabel("Accuracy Type")
plt.ylabel("Average Count")
plt.title("Accuracy Breakdown - Auby")
plt.show()
