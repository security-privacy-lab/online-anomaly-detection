import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score

# Parse the precision/recall text file
file_path = 'darpa_precision_recall.txt'
rows = []
with open(file_path, 'r') as f:
    for line in f:
        match = re.match(r"\[TOP(\d+)\]\s+precision:\s+([\d.]+),\s+recall:\s+([\d.]+)", line.strip())
        if match:
            top = int(match.group(1))
            precision = float(match.group(2))
            recall = float(match.group(3))
            rows.append((top, precision, recall))

df = pd.DataFrame(rows, columns=["Top", "Precision", "Recall"])
print(df.head())

# Plot Precision vs Recall
plt.figure(figsize=(10, 6))
plt.plot(df["Recall"], df["Precision"], marker="o", label="Precision vs Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall for Different Top-N Values")
plt.grid(True)
plt.legend()
plt.show()

# Calculate AUC using Precision–Recall curve
df.sort_values(by="Recall")
pr_auc = auc(df["Recall"], df["Precision"])
print(f"Precision–Recall AUC (from given points): {pr_auc:.4f}")

#This part of the code is for only if you want to have AUC score using anomaly scores and label
'''
file = 'darpa_anomrank.txt'
if os.path.exists(file):
    scores = pd.read_csv(file, sep=' ', header=None, names=['score', 'label'])
    roc_auc = roc_auc_score(scores['label'], scores['score'])
    print(f"ROC-AUC (from scores): {roc_auc:.4f}")
else:
    print(f"File '{file}' not found.")
    '''
