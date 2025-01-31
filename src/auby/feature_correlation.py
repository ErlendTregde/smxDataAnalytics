import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = "./data/auby_latest_200.json"

df = pd.json_normalize(json.load(open(DATA_FILE, "r", encoding="utf-8")))

features = ["score", "max_combo", "perfect1", "perfect2", "misses", "early", "late", "chart.difficulty", "chart.play_count"]
df_corr = df[features].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
