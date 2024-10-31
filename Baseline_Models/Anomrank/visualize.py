# %% [markdown]
# We are going to start by visualizing the Darknet.CSV:
# We first import the darknet.csv using the pandas

# %%
import pandas as pd 
darknet_csv = 'Darknet.CSV'
df = pd.read_csv(darknet_csv)

# %%
# Count how many total Nodes inside of this dataset, and malicious datas:
node_counter = 0 
malicious_node_counter = 0 

for c in df['Labels']:
    if c == 'VPN' or c == 'Tor':
        malicious_node_counter +=1
    node_counter +=1

print(malicious_node_counter, node_counter)
print(node_counter - malicious_node_counter)

# Hence there are 24311 malicous traffics out of total of 158616 traffics 

# %%
node_counter = 0
counter = {}

for c in df['Flow ID']:
    if c not in counter:
        counter[c] = 1
for c in counter.items():
    node_counter +=1
print(node_counter)

# %% [markdown]
# By doing this job, we know how big the dataset is, and how many malicious nodes there should be inside of the dataset. Thereshould be **24311** malicious traffics out of **158616** total of the traffics from the darknet.csv

# %%
df

# %% [markdown]
# Now we are going to sort the dataset by the Source Ip address therefore when we try to visualize the dataset later on, we can visualize it with more connections and relations for each nodes

# %%
# Now we are going to sort the values by the IP addresses not the timestamps therefore we can examine closely to the how the traffics and the dataset works. 
df = df.sort_values(by=['Src IP'])
df

# %%

df['Timestamp'] = pd.to_datetime(df['Timestamp'],errors='coerce')
df


# %% [markdown]
# This is for finding the index for the malicious traffics

# %%
index = 0 
for col in df['Labels']:
    if col == 'VPN' or col == 'Tor':
        print(index)
    index +=1

# %%
import pandas as pd

# Define conditions for malicious and benign traffic
df = df.sort_values(by = ['Src IP'])
malicious_labels = ['VPN', 'Tor']
df['Malicious'] = df['Labels'].apply(lambda x: 1 if x in malicious_labels else 0)

# Group by 'Src IP' to find IPs with both malicious and benign traffic
src_ip_groups = df.groupby('Src IP')['Malicious'].nunique()
src_ips_with_both = src_ip_groups[src_ip_groups > 1].index

# Filter the DataFrame to include only rows where the source IP has both traffic types
df_mixed_traffic = df[df['Src IP'].isin(src_ips_with_both)]

# Display the first few rows of the resulting DataFrame
print(df_mixed_traffic.head())

# Find indices of rows where both malicious and benign traffic exist
mixed_indices = df_mixed_traffic.index.tolist()
print(f"Indices with both malicious and benign traffic: {mixed_indices}")




# %%
import graphviz
# Set parameters
start_row = 116170
end_row = 116190
malicious_labels = ['VPN', 'Tor']
malicious_threshold = 5  # Above this would be considered as malicious
df['Traffic_Type'] = df['Labels'].apply(lambda x: 'malicious' if x in malicious_labels else 'benign')

# Select the range of data
df_range = df.loc[start_row:end_row + 1]

malicious_traffic_count = {}
incoming_malicious_count = {}

dot = graphviz.Digraph(format='png')
dot.attr('node', shape='oval')

# Track malicious traffic counts between source and destination IPs
for _, row in df_range.iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    is_malicious = row['Traffic_Type'] == 'malicious'

    if is_malicious:
        if (src_ip, dst_ip) not in malicious_traffic_count:
            malicious_traffic_count[(src_ip, dst_ip)] = 0
        malicious_traffic_count[(src_ip, dst_ip)] += 1
        if dst_ip not in incoming_malicious_count:
            incoming_malicious_count[dst_ip] = 0
        incoming_malicious_count[dst_ip] += 1



# Add filled nodes for source and destination IPs if they exceed the malicious threshold
for dst_ip, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(dst_ip, color = 'red', style = 'filled')

    # Number of dashed edges (5 malicious packets each)
    dashed_edges = count // 5
    # Number of regular edges (remaining malicious packets)
    regular_edges = count % 5

    # Add dashed edges
    for _ in range(dashed_edges):
        dot.edge(str(src_ip), str(dst_ip), color='red', style='dashed')

    # Add regular edges
    for _ in range(regular_edges):
        dot.edge(str(src_ip), str(dst_ip), color='red')

# Add benign traffic edges
for _, row in df_range[df_range['Traffic_Type'] == 'benign'].iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    dot.edge(str(src_ip), str(dst_ip), color='green')

# Save the visualization
output_path = 'traffic_graph_range'
dot.render(output_path, cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot

# ip connection that has both malicious and benign 

# %% [markdown]
# Now since we have sorted by the source ip address, now we are going to compare the result using the sorting by the timestamps instead of the ip addresses... In the photo above, the dashed lines represents 5 malicious traffics toward the ip address, which was done for the better visualization

# %%
# Now we are going to work on when the dataset has been sorted by the timestamp, instead of the source ip address for comparison 
df = pd.read_csv("Darknet.CSV")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], infer_datetime_format=True, errors='coerce')
df['timestamp_minute'] = df['Timestamp'].dt.floor('min')
df = df.sort_values(by = ['Timestamp'], ascending= True)
df

# %%
# find out which index starts the malicious traffics
index = 0
for c in df['Labels']:
    if c == 'Tor' or c == 'VPN':
        print(index)
    index +=1

# %%
# Set parameters
start_row = 4830
end_row = 4870
malicious_labels = ['VPN', 'Tor']
malicious_threshold = 4  # Above this would be considered as malicious
df['Traffic_Type'] = df['Labels'].apply(lambda x: 'malicious' if x in malicious_labels else 'benign')

# Select the range of data
df_range = df.iloc[start_row:end_row + 1]

malicious_traffic_count = {}
incoming_malicious_count = {}

dot = graphviz.Digraph(format='png')
dot.attr('node', shape='oval')

# Track malicious traffic counts between source and destination IPs
for _, row in df_range.iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    is_malicious = row['Traffic_Type'] == 'malicious'

    if is_malicious:
        if (src_ip, dst_ip) not in malicious_traffic_count:
            malicious_traffic_count[(src_ip, dst_ip)] = 0
        malicious_traffic_count[(src_ip, dst_ip)] += 1
        if dst_ip not in incoming_malicious_count:
            incoming_malicious_count[dst_ip] = 0
        incoming_malicious_count[dst_ip] += 1



# Add filled nodes for source and destination IPs if they exceed the malicious threshold
for dst_ip, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(dst_ip, color = 'red', style = 'filled')

    # Number of dashed edges (5 malicious packets each)
    dashed_edges = count // 5
    # Number of regular edges (remaining malicious packets)
    regular_edges = count % 5

    # Add dashed edges
    for _ in range(dashed_edges):
        dot.edge(str(src_ip), str(dst_ip), color='red', style='dashed')

    # Add regular edges
    for _ in range(regular_edges):
        dot.edge(str(src_ip), str(dst_ip), color='red')

# Add benign traffic edges
for _, row in df_range[df_range['Traffic_Type'] == 'benign'].iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    dot.edge(str(src_ip), str(dst_ip), color='green')

# Save the visualization
output_path = 'traffic_graph_range'
dot.render(output_path, cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot

# timing onto the visualizations, timeframe information onto it; 

# %%
import pandas as pd
import graphviz

# Set parameters
start_row = 4830
end_row = 4870
malicious_labels = ['VPN', 'Tor']
malicious_threshold = 4  # Above this would be considered as malicious

# Identify traffic type
df['Traffic_Type'] = df['Labels'].apply(lambda x: 'malicious' if x in malicious_labels else 'benign')

# Select the range of data
df_range = df.iloc[start_row:end_row + 1]

# Dictionaries to track malicious traffic counts and incoming counts
malicious_traffic_count = {}
incoming_malicious_count = {}

# Initialize Graphviz Digraph
dot = graphviz.Digraph(format='png')
dot.attr('node', shape='oval')

# Track malicious traffic counts and visualize edges with simplified timestamps
for _, row in df_range.iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    timestamp = pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')  # Simplify timestamp format
    is_malicious = row['Traffic_Type'] == 'malicious'

    if is_malicious:
        # Track count of malicious traffic between source and destination
        if (src_ip, dst_ip) not in malicious_traffic_count:
            malicious_traffic_count[(src_ip, dst_ip)] = 0
        malicious_traffic_count[(src_ip, dst_ip)] += 1

        # Track incoming malicious traffic per destination IP
        if dst_ip not in incoming_malicious_count:
            incoming_malicious_count[dst_ip] = 0
        incoming_malicious_count[dst_ip] += 1

        # Visualize malicious edges with reduced timestamp labels in red
        dot.edge(str(src_ip), str(dst_ip), label=timestamp if malicious_traffic_count[(src_ip, dst_ip)] % 3 == 0 else "", color='red')
    else:
        # For benign traffic, add edges with dashed style every 5 packets
        if (src_ip, dst_ip) not in malicious_traffic_count:
            malicious_traffic_count[(src_ip, dst_ip)] = 0
        malicious_traffic_count[(src_ip, dst_ip)] += 1

        # Count for dashed and solid edges, label only occasionally
        count = malicious_traffic_count[(src_ip, dst_ip)]
        style = 'dashed' if count % 5 == 0 else 'solid'
        label = timestamp if count % 5 == 0 else ""
        dot.edge(str(src_ip), str(dst_ip), label=label, color='green', style=style)

# Highlight destination nodes as red if incoming malicious traffic exceeds threshold
for dst_ip, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(dst_ip, color='red', style='filled')

# Save the visualization
output_path = 'traffic_graph_with_timestamps_revised'
dot.render(output_path, cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot


# %%
import pandas as pd
import graphviz

# Set parameters
start_row = 4830
end_row = 4860
malicious_labels = ['VPN', 'Tor']
malicious_threshold = 4  # Above this would be considered as malicious

# Identify traffic type
df['Traffic_Type'] = df['Labels'].apply(lambda x: 'malicious' if x in malicious_labels else 'benign')

# Select the range of data
df_range = df.iloc[start_row:end_row + 1]

# Find source IPs with both malicious and benign traffic
src_ip_groups = df_range.groupby('Src IP')['Traffic_Type'].nunique()
src_ips_with_both_types = src_ip_groups[src_ip_groups > 1].index

# Initialize Graphviz Digraph
dot = graphviz.Digraph(format='png')
dot.attr('node', shape='oval')

# Track incoming malicious traffic count
incoming_malicious_count = {}

# Track malicious and benign traffic with timestamps for relevant IPs
for _, row in df_range.iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    timestamp = pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')  # Simplify timestamp format
    is_malicious = row['Traffic_Type'] == 'malicious'

    # Count incoming malicious traffic for each destination
    if is_malicious:
        if dst_ip not in incoming_malicious_count:
            incoming_malicious_count[dst_ip] = 0
        incoming_malicious_count[dst_ip] += 1

    # Check if the source IP has both types of traffic
    label = timestamp if src_ip in src_ips_with_both_types else ""
    color = 'red' if is_malicious else 'green'
    style = 'dashed' if not is_malicious and row.name % 5 == 0 else 'solid'  # Dashed every 5th benign edge

    # Add edge with timestamp label only for IPs with both traffic types
    dot.edge(str(src_ip), str(dst_ip), label=label, color=color, style=style)

# Highlight destination nodes if they exceed the malicious threshold
for dst_ip, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(dst_ip, color='red', style='filled')

# Save the visualization
output_path = 'traffic_graph_mixed_timestamps'
dot.render(output_path, cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot


# %%
import pandas as pd
import graphviz

# Set parameters
start_row = 4830
end_row = 4870
malicious_labels = ['VPN', 'Tor']
malicious_threshold = 4  # Above this would be considered as malicious

# Identify traffic type
df['Traffic_Type'] = df['Labels'].apply(lambda x: 'malicious' if x in malicious_labels else 'benign')

# Select the range of data
df_range = df.iloc[start_row:end_row + 1]

# Find source IPs with both malicious and benign traffic
src_ip_groups = df_range.groupby('Src IP')['Traffic_Type'].nunique()
src_ips_with_both_types = src_ip_groups[src_ip_groups > 1].index

# Initialize Graphviz Digraph
dot = graphviz.Digraph(format='png')
dot.attr('node', shape='oval')

# Track incoming malicious traffic count
incoming_malicious_count = {}

# Track malicious and benign traffic with colored timestamps for relevant IPs
for _, row in df_range.iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    timestamp = pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')  # Simplify timestamp format
    is_malicious = row['Traffic_Type'] == 'malicious'

    # Count incoming malicious traffic for each destination
    if is_malicious:
        if dst_ip not in incoming_malicious_count:
            incoming_malicious_count[dst_ip] = 0
        incoming_malicious_count[dst_ip] += 1

    # Check if the source IP has both types of traffic and set label color
    if src_ip in src_ips_with_both_types:
        label_color = 'red' if is_malicious else 'green'
        label = f'<<FONT COLOR="{label_color}">{timestamp}</FONT>>'  # Enclose label in double angle brackets
    else:
        label = ""  # No label for edges without mixed traffic

    # Set edge color and style
    color = 'red' if is_malicious else 'green'
    style = 'dashed' if not is_malicious and row.name % 5 == 0 else 'solid'  # Dashed every 5th benign edge

    # Add edge with colored timestamp label only for IPs with both traffic types
    dot.edge(str(src_ip), str(dst_ip), label=label, color=color, style=style)

# Highlight destination nodes if they exceed the malicious threshold
for dst_ip, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(dst_ip, color='red', style='filled')

# Save the visualization
output_path = 'traffic_graph_mixed_colored_timestamps'
dot.render(output_path, cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot


# %%
import pandas as pd
from IPython.display import HTML, display

# Set parameters
start_row = 4830
end_row = 4870
malicious_labels = ['VPN', 'Tor']

# Identify traffic type
df['Traffic_Type'] = df['Labels'].apply(lambda x: 'malicious' if x in malicious_labels else 'benign')

# Select the range of data
df_range = df.iloc[start_row:end_row + 1]

# Sort by timestamp for ordered display
df_range['Timestamp'] = pd.to_datetime(df_range['Timestamp'])
df_range = df_range.sort_values(by='Timestamp')

# Find source IPs with both malicious and benign traffic
src_ip_groups = df_range.groupby('Src IP')['Traffic_Type'].nunique()
src_ips_with_both_types = src_ip_groups[src_ip_groups > 1].index

# Filter rows with source IPs that have both malicious and benign traffic
df_mixed_traffic = df_range[df_range['Src IP'].isin(src_ips_with_both_types)]

# Create HTML table
table_html = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
table_html += "<tr><th>Timestamp</th><th>Source IP</th><th>Destination IP</th><th>Traffic Type</th></tr>"

# Populate table with rows
for _, row in df_mixed_traffic.iterrows():
    src_ip = row['Src IP']
    dst_ip = row['Dst IP']
    timestamp = row['Timestamp'].strftime('%Y-%m-%d %H:%M')
    traffic_type = row['Traffic_Type']
    
    # Set font color based on traffic type
    color = 'red' if traffic_type == 'malicious' else 'green'
    traffic_type_display = f"<span style='color:{color};'>{traffic_type.capitalize()}</span>"
    
    # Add row to table
    table_html += f"<tr><td>{timestamp}</td><td>{src_ip}</td><td>{dst_ip}</td><td>{traffic_type_display}</td></tr>"

table_html += "</table>"

# Display the table
display(HTML(table_html))


# %% [markdown]
# Now lets start working on the darpa dataset visualization!

# %%
import pandas as pd 
column_names = ['Timestamp', 'Src IP', 'Dst IP', 'Label']
df = pd.read_csv('darpa.txt', names = column_names, header = None, delimiter= ' ')
df

# %% [markdown]
# Same as we did to the darknet.csv, we are going to sort the dataset by the source Ip addresses

# %%
df.sort_values(by='Src IP', ascending= True)

# %% [markdown]
# We are going to find where the malicious traffic is going to start for visualization purpose

# %%
index = 0
for col in df['Label']:
    if col == 1:
        print(index)
    index +=1 

# %%
import graphviz as viz

# Subset of the DataFrame
df_subset = df.iloc[66489:66530]
malicious_threshold = 5

dot = viz.Digraph(format='png')
dot.attr('node', shape='oval')

# Dictionary to count incoming malicious traffic for each destination
incoming_malicious_count = {}

# Iterate through each row of the subset
for index, row in df_subset.iterrows():
    source = str(row['Src IP'])
    destination = str(row['Dst IP'])
    traffic_type = row['Label']

    if traffic_type == 1:
        # Count incoming malicious traffic for the destination
        if destination not in incoming_malicious_count:
            incoming_malicious_count[destination] = 0
        incoming_malicious_count[destination] += 1

        # Add edge with red color
        dot.edge(source, destination, color='red')
    else:
        # Add edge with green color for benign traffic
        dot.edge(source, destination, color='green')

# Mark nodes as malicious if they receive more than the threshold
for destination, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(destination, color='red', style='filled')

# Save the visualization
output_path = 'network_graph'
dot.render(output_path, format='png', cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot

# Ip addresses
'''DARPA [16] has 4.5M IP-IP communications between 9.4K source
IP and 2.3K destination IP over 87.7K minutes. Each communication
is a directed edge (srcIP, dstIP, timestamp, attack) where the attack
label indicates whether the communication is an attack or not.'''

# %%
df = df.sort_values(by = ['Timestamp'], ascending= True)
df

# %%
index = 0
for col in df['Label']:
    if col == 1:
        print(index)
    index +=1 

# %%
import graphviz as viz

# Subset of the DataFrame
df_subset = df.iloc[66500:66515]
malicious_threshold = 5

dot = viz.Digraph(format='png')
dot.attr('node', shape='oval')

# Dictionary to count incoming malicious traffic for each destination
incoming_malicious_count = {}

# Iterate through each row of the subset
for index, row in df_subset.iterrows():
    source = str(row['Src IP'])
    destination = str(row['Dst IP'])
    traffic_type = row['Label']

    if traffic_type == 1:
        # Count incoming malicious traffic for the destination
        if destination not in incoming_malicious_count:
            incoming_malicious_count[destination] = 0
        incoming_malicious_count[destination] += 1

        # Add edge with red color
        dot.edge(source, destination, color='red')
    else:
        # Add edge with green color for benign traffic
        dot.edge(source, destination, color='green')

# Mark nodes as malicious if they receive more than the threshold
for destination, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(destination, color='red', style='filled')

# Save the visualization
output_path = 'network_graph'
dot.render(output_path, format='png', cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot

# work on the ip address for the darpa 

# %% [markdown]
# Now we are going to work on the ROC curves using the AnomRank(for now)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load data from darpa_anomrank.txt which is the result from running the anomrank
file_path = 'darpa_anomrank_for_darpa.txt'
scores = []
labels = []

# Read the file and separate scores and labels
with open(file_path, 'r') as f:
    for line in f:
        score, label = map(float, line.strip().split())
        scores.append(score)
        labels.append(int(label))

# Calculate FPR, TPR, and thresholds using sklearn's roc_curve
fpr, tpr, thresholds = roc_curve(labels, scores)

# Calculate the AUC (Area Under Curve)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Reference Line')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for DARPA Dataset')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# %%
import matplotlib.pyplot as plt
import pandas as pd

# Read the precision and recall data
data = pd.read_csv('precision_recall_data_for_darpa.txt', sep=' ', header=None, names=['Top', 'Precision', 'Recall'])

# Plot Precision vs Recall
plt.figure(figsize=(10, 6))
plt.plot(data['Recall'], data['Precision'], marker='o', label='Precision vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall')
plt.grid(True)
plt.show()


# %% [markdown]
# Now for the generating the darknet.csv for the anomrank format.(preprocess)

# %%
import hashlib
df = pd.read_csv('Darknet.CSV')
df

# %%
import hashlib
import pandas as pd

def hash_ip(ip):
    """Hash IP addresses to integers within a smaller range to reduce node size."""
    ip_str = str(ip)  # Ensure the IP is a string
    return int(hashlib.md5(ip_str.encode()).hexdigest(), 16) % (10**6)  # Reducing the range to 6 digits

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])

# Convert 'Timestamp' to integer minutes since epoch
df["timestamp_minute"] = (df["Timestamp"].view('int64') // 10**9) // 60

# Ensure IPs are strings before hashing
df['Src IP'] = df['Src IP'].astype(str).apply(hash_ip)
df['Dst IP'] = df['Dst IP'].astype(str).apply(hash_ip)

# Categorize labels into anomalies and benign
df['Label Condition'] = df['Labels'].apply(lambda x: 0 if x.lower() in ["nonvpn", "non-tor"] else 1)

# Print anomaly and benign counts
print(f"Total anomalies: {sum(df['Label Condition'] == 1)}")
print(f"Total benign instances: {sum(df['Label Condition'] == 0)}")


# Prepare data for AnomRank
anomrank_data = df[['timestamp_minute', 'Src IP', 'Dst IP', 'Label Condition']]
anomrank_data.to_csv('traffic.txt', sep=' ', index=False, header=False)

print(f"Data has been formatted and saved to traffic.txt")


# %%
node_counter = 0 
for c in df['Label']:
    node_counter +=1
print(node_counter)

# %%
column_names = ['Timestamp', 'Src IP', 'Dst IP', 'Label']
df.sort_values(by='Src IP', ascending= True)
df = pd.read_csv('traffic.txt', names = column_names, header = None, delimiter= ' ')
df

# %%
index = 0
for i, c in enumerate(df['Label']):  # Use enumerate for clarity
    if c == 1:
        print(i)


# %%
import graphviz as viz

# Subset of the DataFrame
df_subset = df.loc[134290:134320]
malicious_threshold = 5

dot = viz.Digraph(format='png')
dot.attr('node', shape='oval')

# Dictionary to count incoming malicious traffic for each destination
incoming_malicious_count = {}

# Iterate through each row of the subset
for index, row in df_subset.iterrows():
    source = str(row['Src IP'])
    destination = str(row['Dst IP'])
    traffic_type = row['Label']

    if traffic_type == 1:
        # Count incoming malicious traffic for the destination
        if destination not in incoming_malicious_count:
            incoming_malicious_count[destination] = 0
        incoming_malicious_count[destination] += 1

        # Add edge with red color
        dot.edge(source, destination, color='red')
    else:
        # Add edge with green color for benign traffic
        dot.edge(source, destination, color='green')

# Mark nodes as malicious if they receive more than the threshold
for destination, count in incoming_malicious_count.items():
    if count > malicious_threshold:
        dot.node(destination, color='red', style='filled')

# Save the visualization
output_path = 'network_graph'
dot.render(output_path, format='png', cleanup=True)
print(f"Visualization saved as {output_path}.png")
dot

# Make a cversion with timestamps


# %% [markdown]
# Since we have preprocessed the dataset, we are going to run this dataset to the anomrank and receive the results..:

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Read the precision and recall data
data = pd.read_csv('precision_recall_data_for_darknet.txt', sep=' ', header=None, names=['Top', 'Precision', 'Recall'])

# Plot Precision vs Recall
plt.figure(figsize=(10, 6))
plt.plot(data['Recall'], data['Precision'], marker='o', label='Precision vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall')
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load data from darpa_anomrank.txt
file_path = 'darpa_anomrank_for_darknet_csv2.txt'
scores = []
labels = []

# Read the file and separate scores and labels
with open(file_path, 'r') as f:
    for line in f:
        score, label = map(float, line.strip().split())
        scores.append(score)
        labels.append(int(label))

# Calculate FPR, TPR, and thresholds using sklearn's roc_curve
fpr, tpr, thresholds = roc_curve(labels, scores)

# Calculate the AUC (Area Under Curve)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Reference Line')

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for DARPA Dataset')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# %% [markdown]
# ### Explanation:
# 
# First, the code reads the data from the darpa_anomrank.txt file which contains two values: the anomaly score(predicted by the AnomRank) and actual labels(from the ground value)
# - Therefore the first column represents the anomaly score, which is calculated by the anomrank, which indicates the likelihood of a traffic being anomalous(higher value means higher suspicision..)
# - Second column is label, indicating the ground value, which indicates whether the value was truely malicious or not. 
# 
# 
# Scond, now it starts to calculate the ROC curve using the sklearn.metrics. It calculates the FPR and TPR and the thresholds that separates benign from amlicious traffic. ROC curve plots TPR vs. FPR at different thresholds
# - TPR is also known as Recall, therefore for the TPR, I have used recall value. 
# 
# Third, now it calculates the AUC level, using the function auc()
# 
# After doing all of these calculations I started to plot them into graph formatting therefore we can visualize the dataset.
# According to the dataset that I have plotted, it seems like Darknet.csv seems like to have a imbalance between the benign and malicious traffics. 
# 
# 
# #### ROC curve being calculated:
# Iterate Through Each Threshold:
# 
# - For each threshold, compute TPR and FPR by comparing the anomaly scores to the threshold and comparing the predicted classification to the true labels.
# - The threshold determines the decision boundary for labeling a data point as either benign or malicious.
# 
# Calculate TPR and FPR:
# - At each threshold, update the confusion matrix to get the current TP, FP, TN, and FN counts.
# 
# 
# ![image.png](attachment:image.png)


