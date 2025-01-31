import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = "./data/auby_latest_200.json"

df = pd.json_normalize(json.load(open(DATA_FILE, "r", encoding="utf-8")))

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["chart.difficulty"], y=df["score"])
plt.xlabel("Difficulty")
plt.ylabel("Score")
plt.title("Impact of Difficulty on Score")
plt.show()
