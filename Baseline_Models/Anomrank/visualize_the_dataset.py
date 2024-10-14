'''import pandas as pd
import matplotlib.pyplot as plt

def hash_ip(ip):
    """Hash IP addresses to integers within a smaller range to reduce node size."""
    import hashlib
    return int(hashlib.md5(ip.encode()).hexdigest(), 16) % (10**6)  # Reducing the range to 6 digits

# Read the dataset
input_file = "C:/Users/seoji/Desktop/Codes/sedanspot/Darknet.CSV"  # Change this to the path of your dataset
df = pd.read_csv(input_file)

# Convert 'Timestamp' to datetime format and filter out invalid timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Create new minute-based timestamp column (converting to Unix time in minutes)
df["timestamp_minute"] = (df["Timestamp"].astype(int) // 10**9) // 60

# Hash the IP addresses into integers (ensure no duplicates or malformed IPs)
df['Src IP'] = df['Src IP'].apply(hash_ip)
df['Dst IP'] = df['Dst IP'].apply(hash_ip)

# Remove rows with identical source and destination IPs (optional for cleaning)
df = df[df['Src IP'] != df['Dst IP']]

# Convert labels to 0 (benign) or 1 (malicious)
df['Label Condition'] = df['Label'].apply(lambda x: 0 if x.lower() in ["nonvpn", "non-tor"] else 1)

# Plotting the data
plt.figure(figsize=(12, 8))

# Plot benign traffic in black
benign = df[df['Label Condition'] == 0]
plt.scatter(benign['timestamp_minute'], benign['Src IP'], c='black', s=1, label='Benign Traffic', alpha=0.6)

# Plot malicious traffic in red
malicious = df[df['Label Condition'] == 1]
plt.scatter(malicious['timestamp_minute'], malicious['Src IP'], c='red', s=1, label='Malicious Traffic', alpha=0.6)

# Customize the plot
plt.xlabel('Time (Minutes)')
plt.ylabel('Hashed Source IP')
plt.title('Traffic Patterns: Malicious (Red) vs. Benign (Black)')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

def hash_ip(ip):
    """Hash IP addresses to integers within a smaller range to reduce node size."""
    return int(hashlib.md5(ip.encode()).hexdigest(), 16) % (10**6)  # Reducing the range to 6 digits

# Read the dataset
input_file = "path/to/your/Darknet.CSV"  # Change this to the path of your dataset
df = pd.read_csv(input_file)

# Convert 'Timestamp' to datetime format and filter out invalid timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Hash the IP addresses into integers
df['Src IP'] = df['Src IP'].apply(hash_ip)
df['Dst IP'] = df['Dst IP'].apply(hash_ip)

# Convert labels to 0 (benign) or 1 (malicious)
df['Label Condition'] = df['Label'].apply(lambda x: 0 if x.lower() in ["nonvpn", "non-tor"] else 1)

# Create a new column for the date and hour (useful for daily/hourly aggregation)
df['date_hour'] = df['Timestamp'].dt.floor('H')

# Aggregate the counts of benign and malicious traffic per hour
time_series_data = df.groupby(['date_hour', 'Label Condition']).size().unstack(fill_value=0)
time_series_data.columns = ['Benign', 'Malicious']

# Plot the time series data
plt.figure(figsize=(14, 6))
plt.plot(time_series_data.index, time_series_data['Benign'], label='Benign Traffic', color='black', alpha=0.6)
plt.plot(time_series_data.index, time_series_data['Malicious'], label='Malicious Traffic', color='red', alpha=0.6)
plt.fill_between(time_series_data.index, time_series_data['Benign'], color='black', alpha=0.2)
plt.fill_between(time_series_data.index, time_series_data['Malicious'], color='red', alpha=0.2)
plt.xlabel('Time (Hourly)')
plt.ylabel('Number of Events')
plt.title('Benign vs. Malicious Traffic Over Time')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

# Create a heatmap for daily activity (optional)
df['hour'] = df['Timestamp'].dt.hour
df['day'] = df['Timestamp'].dt.date

heatmap_data = df.pivot_table(index='hour', columns='day', values='Label Condition', aggfunc='sum', fill_value=0)
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap='Reds', linewidths=0.5, linecolor='black')
plt.title('Heatmap of Malicious Traffic by Hour and Day')
plt.xlabel('Day')
plt.ylabel('Hour')
plt.tight_layout()
plt.show()
