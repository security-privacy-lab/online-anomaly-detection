import os
import pandas as pd
from sklearn.metrics import roc_auc_score


# change here
file = 'user12_anomrank_combined_or.txt'
if not os.path.exists(file):
    print(f"File '{file}' not found.")
else:
    df = pd.read_csv(file, sep=' ', header=None, names=['score', 'label'])
    auc = roc_auc_score(df['label'], df['score'])
    print(f"AUC: {auc:.4f}")
