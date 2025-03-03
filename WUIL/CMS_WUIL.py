import math, random, collections
from datetime import datetime
class CountMinSketch:

    def __init__(self, epsilon=0.001, delta=0.01):
        self.epsilon = epsilon
        self.delta = delta
        self.width = int(math.ceil(math.e / epsilon))
        self.depth = int(math.ceil(math.log(1.0/delta, 2)))
        self.table = [[0] * self.width for _ in range(self.depth)]
        self.hash_seeds = [random.randint(1, 1000000) for _ in range(self.depth)]
    
    def _hash(self, item, i):
        return (hash(str(item)) ^ self.hash_seeds[i]) % self.width
    
    def update(self, item, count=1):
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i][idx] += count
    
    def query(self, item):
        estimates = []
        for i in range(self.depth):
            idx = self._hash(item, i)
            estimates.append(self.table[i][idx])
        return min(estimates)

##############################
# Parsing function for the dataset
##############################

def parse_dataset_line(line):

    # Skip header lines
    if line.lower().startswith("src_node") or not line.strip():
        return None
    parts = line.strip().split('|')
    if len(parts) < 4:
        return None
    src_chain = parts[0].strip()      # e.g., "0\1\2\3\4"
    dst_chain = parts[1].strip()      # e.g., "0\1\2\3\4" or "0\1\2\3\7\4"
    timestamp_str = parts[2].strip()  # e.g., "35396524"
    label_str = parts[3].strip()
    
    try:
        timestamp = float(timestamp_str)
    except ValueError:
        print(f"Error converting timestamp: {timestamp_str}")
        return None
    try:
        label = int(label_str)
    except ValueError:
        label = 0

    # Compute source chain info.
    src_nodes = [token for token in src_chain.split('\\') if token]
    if not src_nodes:
        return None
    src_depth = len(src_nodes)
    src_end = src_nodes[-1]
    
    # Compute destination chain info.
    dst_nodes = [token for token in dst_chain.split('\\') if token]
    if not dst_nodes:
        return None
    dst_depth = len(dst_nodes)
    dst_end = dst_nodes[-1]
    
    # Define a combined depth (average of the two depths)
    combined_depth = (src_depth + dst_depth) / 2.0
    
    # Define a combined end_node as a tuple.
    combined_end = (src_end, dst_end)
    
    return timestamp, combined_depth, combined_end, label


def process_dataset_cms_fixed_window(filename, window_size=10, epsilon=0.001, delta=0.01):

    cms = CountMinSketch(epsilon, delta)
    window = collections.deque()  # will store tuples: (timestamp, combined_depth, label, combined_end)
    rolling_depth_sum = 0.0

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_dataset_line(line)
            if parsed is None:
                continue
            timestamp, combined_depth, combined_end, label = parsed
            
            # If window is full, remove the oldest record.
            if len(window) == window_size:
                oldest = window.popleft()
                oldest_timestamp, oldest_depth, oldest_label, oldest_combined_end = oldest
                cms.update(oldest_combined_end, -1)
                rolling_depth_sum -= oldest_depth
            
            # Add new record.
            window.append((timestamp, combined_depth, label, combined_end))
            cms.update(combined_end, 1)
            rolling_depth_sum += combined_depth
            
            # Compute rolling average combined depth.
            avg_depth = rolling_depth_sum / len(window)
            
            # Compute rolling average inter-arrival time.
            if len(window) > 1:
                timestamps = [r[0] for r in window]
                diffs = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
                avg_interarrival = sum(diffs) / len(diffs)
            else:
                avg_interarrival = None
            
        
            approx_freq = cms.query(combined_end)
            
            print(f"Line: {line.strip()}")
            print(f"  -> Timestamp: {timestamp:.2f}")
            print(f"  -> Combined Depth (avg of src & dst depths): {combined_depth:.2f}")
            print(f"  -> Combined end node: {combined_end}")
            print(f"  -> Rolling average combined depth (last {len(window)} records): {avg_depth:.2f}")
            if avg_interarrival is not None:
                print(f"  -> Rolling average inter-arrival time: {avg_interarrival:.2f} s")
            else:
                print("  -> Not enough data for inter-arrival time yet.")
            print(f"  -> Approx frequency of combined end node {combined_end} in window: {approx_freq}")
            print(f"  -> Label: {label}\n")

if __name__ == "__main__":
    # Replace 'dataset.txt' with your dataset file path.
    process_dataset_cms_fixed_window('testtest..csv', window_size=10, epsilon=0.001, delta=0.01)
