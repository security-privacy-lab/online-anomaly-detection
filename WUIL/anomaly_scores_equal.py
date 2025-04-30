# SECOND EXAMPLE

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# -------------------------------
# CONFIGURATION
# -------------------------------
WINDOW_SIZE = 800
WINDOW_STRIDE = 200
MEMORY_DIM = 3
DECAY_TUNED = 0.5

# -------------------------------
# LOAD & PREPROCESS DATA
# -------------------------------
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

# -------------------------------
# MEMORY-BASED SCORING
# -------------------------------
def initialize_memory_vector(edge_data, stats):
    vec = np.zeros(MEMORY_DIM)
    vec[0] = edge_data['depth'] / 10.0
    vec[1] = edge_data['time_since_last'] / 10000.0
    vec[2] = (edge_data['degree'] - stats['mean_degree']) / stats['std_degree']
    return vec

def update_memory_tuned(memory, node, new_vec, activity_counter):
    base_decay = DECAY_TUNED
    activity = activity_counter[node]
    decay = base_decay * (1 - 0.3 * (activity / (activity + 10)))
    if node not in memory:
        memory[node] = new_vec
    else:
        memory[node] = decay * memory[node] + (1 - decay) * new_vec
    activity_counter[node] += 1
    return memory[node]

def compute_window_memory_snapshot_tuned(edges, memory, activity_counter, stats):
    snapshot = {}
    for edge in edges:
        for node in [edge['src'], edge['dst']]:
            vec = initialize_memory_vector(edge, stats)
            updated = update_memory_tuned(memory, node, vec, activity_counter)
            snapshot[node] = np.copy(updated)
    return snapshot

def calculate_window_score(current_snap, previous_snap):
    diffs = []
    for node in current_snap:
        if node in previous_snap:
            diffs.append(np.linalg.norm(current_snap[node] - previous_snap[node]))
    if not diffs:
        return 0
    return 0.5 * np.percentile(diffs, 75) + 0.3 * np.median(diffs) + 0.2 * np.mean(diffs)

def run_unsupervised_anomaly_detection(edges, stats, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE, lag=2):
    memory = defaultdict(lambda: np.zeros(MEMORY_DIM))
    activity_counter = defaultdict(int)
    snapshots = []
    scores = []
    labels = []

    for i in range(0, len(edges) - window_size + 1, stride):
        window = edges[i:i + window_size]
        label_vals = [e['label'] for e in window]
        current_snapshot = compute_window_memory_snapshot_tuned(window, memory, activity_counter, stats)
        if len(snapshots) < lag:
            score = 0
        else:
            score = calculate_window_score(current_snapshot, snapshots[-lag])
        scores.append(score)
        labels.append(1 if sum(label_vals) > window_size // 2 else 0)
        snapshots.append(current_snapshot)

    return scores, labels

# -------------------------------
# SECOND DERIVATIVE + HYBRID (NO IF)
# -------------------------------
def compute_second_derivative_scores(base_scores):
    base_scores = np.array(base_scores)
    second_deriv = np.zeros_like(base_scores)
    for t in range(2, len(base_scores)):
        diff1 = base_scores[t] - base_scores[t - 1]
        diff2 = base_scores[t - 1] - base_scores[t - 2]
        second_deriv[t] = abs(diff1 - diff2)
    if np.max(second_deriv) != 0:
        second_deriv = second_deriv / (np.max(second_deriv) + 1e-9)
    return second_deriv

def compute_hybrid_scores_no_if(base_scores, weight_base=0.85, weight_deriv2=0.15):
    second_deriv_scores = compute_second_derivative_scores(base_scores)
    hybrid = (weight_base * np.array(base_scores) +
              weight_deriv2 * second_deriv_scores)
    return hybrid, second_deriv_scores

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    filename = "custom_data/TESTO.csv"
    edges = load_edges(filename)
    stats = compute_dataset_stats(edges)

    scores, labels = run_unsupervised_anomaly_detection(edges, stats)
    hybrid_scores, deriv2 = compute_hybrid_scores_no_if(scores)

    auc = roc_auc_score(labels, hybrid_scores)
    print(f"Hybrid ROC AUC with second derivative: {auc:.4f}")

# Base score(run_unsupervised_anomaly_detection(edges, stats)) = how different the graph's node behavior is now compared to the past.
# This is a hyperparameter you define (default = 0.85) to determine how much influence the base score has in the final hybrid anomaly score.