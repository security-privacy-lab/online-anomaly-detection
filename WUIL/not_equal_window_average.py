#### FIRST EXAMPLE 

import numpy as np
import matplotlib.pyplot as plt
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

def compute_edge_scores(edges, stats):
    memory = defaultdict(lambda: np.zeros(MEMORY_DIM))
    activity_counter = defaultdict(int)
    scores = []
    labels = []

    for edge in edges:
        vecs = []
        for node in [edge['src'], edge['dst']]:
            vec = initialize_memory_vector(edge, stats)
            updated_vec = update_memory(memory, node, vec, activity_counter)
            vecs.append(np.copy(updated_vec))

        # L2 norm between memory vectors this gives us numeric representation how much node's behavior has changed across the time
        # If node vector changes a lot, the memory vector "jumps" meaning higher L2 norm -> suspicious 
        
        score = np.linalg.norm(vecs[0] - vecs[1])
        scores.append(score)
        labels.append(edge['label'])

    return scores, labels

def compute_second_derivative(scores):
    scores = np.array(scores)
    second_deriv = np.zeros_like(scores)
    for i in range(2, len(scores)):
        diff1 = scores[i] - scores[i - 1]
        diff2 = scores[i - 1] - scores[i - 2]
        second_deriv[i] = abs(diff1 - diff2)
    if np.max(second_deriv) != 0:
        second_deriv /= (np.max(second_deriv) + 1e-9)
    return second_deriv

# RUN
filename = "custom_data/TESTO.csv"
edges = load_edges(filename)
stats = compute_dataset_stats(edges)
base_scores, labels = compute_edge_scores(edges, stats)
second_deriv_scores = compute_second_derivative(base_scores)
hybrid_scores = 0.85 * np.array(base_scores) + 0.15 * second_deriv_scores
auc = roc_auc_score(labels, hybrid_scores)
print("Hybrid ROC AUC with second derivative:", auc)
