#### THIRD EXAMPLE 
# Does simple edge-level anomaly detection using node memory dynamics in the graphs. Detects the anomalous behavior between nodes as it tracks the updated memory vectors for each node
# Calculate how much their behavior diverges over time, assign a score to each edge and higher score is suspicious interaction 
# This cod uses normalized depth difference, normalized time lag, Z-score degree and update the memory for each node and compute the edge anomaly scores using L2 norm
# Difference between the first example and this one is that this one separately updates the src and dst features and memory vectors are updated and then compared 
# Does not use second deriative or hybridization, just L2 difference 

# The input format is : src,dst,time,label


import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

MEMORY_DIM = 3
DECAY_TUNED = 0.5

def load_edges(filename):
    edges = []
    last_seen = {}
    degree_counter = defaultdict(int)

    with open(filename, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            if len(data) != 4:
                continue
            src, dst, timestamp, label = data
            timestamp = int(timestamp)
            label = int(label)

            src_depth = len(Path(src).parts)
            dst_depth = len(Path(dst).parts)
            depth = abs(src_depth - dst_depth)

            prev_time = last_seen.get(dst, timestamp)
            time_since_last = timestamp - prev_time
            last_seen[dst] = timestamp

            degree_counter[src] += 1
            degree_counter[dst] += 1

            edges.append({
                'src': src,
                'dst': dst,
                'depth': depth,
                'time_since_last': time_since_last,
                'degree': degree_counter[dst],
                'label': label
            })
    return edges

def compute_dataset_stats(edges):
    degrees = [e['degree'] for e in edges]
    return {
        'mean_degree': np.mean(degrees),
        'std_degree': np.std(degrees) + 1e-9
    }

def initialize_memory_vector(edge_data, stats):
    vec = np.zeros(MEMORY_DIM)
    vec[0] = edge_data['depth'] / 10.0
    vec[1] = edge_data['time_since_last'] / 10000.0
    vec[2] = (edge_data['degree'] - stats['mean_degree']) / stats['std_degree']
    return vec

def update_memory(memory, node, new_vec, activity_counter):
    base_decay = DECAY_TUNED
    activity = activity_counter[node]
    decay = base_decay * (1 - 0.3 * (activity / (activity + 10)))
    if node not in memory:
        memory[node] = new_vec
    else:
        memory[node] = decay * memory[node] + (1 - decay) * new_vec
    activity_counter[node] += 1
    return memory[node]

def detect_edge_anomalies(edges, stats):
    memory = defaultdict(lambda: np.zeros(MEMORY_DIM))
    activity_counter = defaultdict(int)
    scores = []
    labels = []

    for edge in edges:
        vec_src = initialize_memory_vector(edge, stats)
        vec_dst = initialize_memory_vector(edge, stats)

        updated_src = update_memory(memory, edge['src'], vec_src, activity_counter)
        updated_dst = update_memory(memory, edge['dst'], vec_dst, activity_counter)

        score = np.linalg.norm(updated_src - updated_dst)
        scores.append(score)
        labels.append(edge['label'])

    return scores, labels

if __name__ == "__main__":
    filename = "custom_data/TESTO.csv"  # or wherever your CSV is
    edges = load_edges(filename)
    stats = compute_dataset_stats(edges)
    scores, labels = detect_edge_anomalies(edges, stats)

    auc = roc_auc_score(labels, scores)
    print(f"Edge-Level ROC AUC: {auc:.4f}")

# Check if the move is very similar to the past move. Certain time frames. How can we describe them in edge level
# Any way to get the score updates for certain move of the past.  -> Use Window base but should not be fixed lengths and vary the size 
# Not a single edge, but consider the path scores
# If we describe the node behavior
