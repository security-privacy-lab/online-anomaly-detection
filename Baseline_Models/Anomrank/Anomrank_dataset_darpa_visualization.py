import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the DARPA dataset
input_file = "C:/Users/seoji/Desktop\Codes/KDD19-AnomRank-master/darpa.txt"  # Replace with the path to your DARPA dataset file
columns = ['time_minute', 'Src IP', 'Dst IP', 'Label Condition']
df = pd.read_csv(input_file, sep=' ', names=columns)

# Convert 'time_minute' to a datetime-like format if needed (otherwise keep as integer minutes)
df['time_minute'] = pd.to_numeric(df['time_minute'], errors='coerce')
df = df.dropna(subset=['time_minute'])

# Display the number of anomalies and benign instances
print(f"Total anomalies: {sum(df['Label Condition'] == 1)}")
print(f"Total benign instances: {sum(df['Label Condition'] == 0)}")

# Plot a time series of malicious events using a rolling window (e.g., 10-minute average)
rolling_window = 10
malicious_trend = df[df['Label Condition'] == 1].groupby('time_minute').size().rolling(rolling_window).mean()
plt.figure(figsize=(12, 6))
plt.plot(malicious_trend.index, malicious_trend.values, label='Malicious Trend (10-min average)', color='red', alpha=0.7)
plt.title('Trend of Malicious Traffic Over Time')
plt.xlabel('Time (Minutes)')
plt.ylabel('Number of Malicious Events')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Density plot to show the concentration of malicious traffic over time
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Label Condition'] == 1]['time_minute'], bins=50, kde=True, color='red', alpha=0.6)
plt.title('Density of Malicious Traffic Over Time')
plt.xlabel('Time (Minutes)')
plt.ylabel('Frequency of Malicious Events')
plt.tight_layout()
plt.show()

# Scatter plot: Visualize malicious events (red) and benign events (black)
plt.figure(figsize=(14, 7))
plt.scatter(df[df['Label Condition'] == 1]['time_minute'], df[df['Label Condition'] == 1]['Src IP'],
            color='red', s=10, label='Malicious Traffic', alpha=0.5)
plt.scatter(df[df['Label Condition'] == 0]['time_minute'], df[df['Label Condition'] == 0]['Src IP'],
            color='black', s=1, label='Benign Traffic', alpha=0.3)
plt.title('Scatter Plot of Traffic Patterns: Malicious (Red) vs. Benign (Black)')
plt.xlabel('Time (Minutes)')
plt.ylabel('Source IP (Hashed)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
