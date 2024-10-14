import matplotlib.pyplot as plt
import pandas as pd

# Read the precision and recall data
data = pd.read_csv('C:/Users/seoji/Desktop/Codes/KDD19-AnomRank-master/precision_recall_data.txt', sep=' ', header=None, names=['Top', 'Precision', 'Recall'])

# Plot Precision vs Recall
plt.figure(figsize=(10, 6))
plt.plot(data['Recall'], data['Precision'], marker='o', label='Precision vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall for Different Top 250 Values')
plt.grid(True)
plt.show()
