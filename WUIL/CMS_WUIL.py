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
    """
    Parses a line from the dataset file in the format:
       src_node|dst_node|timestamp|label
       

    """
    # Skip header lines
    if line.lower().startswith("src_node") or not line.strip():
        return None
    parts = line.strip().split('|')
    if len(parts) < 4:
        return None
    src_chain = parts[0].strip()  # e.g., "0\1\2\3\4"
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
    
    src_nodes = [token for token in src_chain.split('\\') if token]
    if not src_nodes:
        return None
    depth = len(src_nodes)
    end_node = src_nodes[-1]
    
    return timestamp, depth, end_node, label


def process_dataset_cms_fixed_window(filename, window_size=10, epsilon=0.001, delta=0.01):
    cms = CountMinSketch(epsilon, delta)
    window = collections.deque()  # will store tuples: (timestamp, depth, label, end_node)
    rolling_depth_sum = 0.0

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_dataset_line(line)
            if parsed is None:
                continue
            timestamp, depth, end_node, label = parsed
            
            # If window is full, remove the oldest record and update CMS negatively.
            if len(window) == window_size:
                oldest = window.popleft()
                oldest_timestamp, oldest_depth, oldest_label, oldest_end_node = oldest
                cms.update(oldest_end_node, -1)
                rolling_depth_sum -= oldest_depth
            
            # Add the new record.
            window.append((timestamp, depth, label, end_node))
            cms.update(end_node, 1)
            rolling_depth_sum += depth
            
            # Compute rolling average depth.
            avg_depth = rolling_depth_sum / len(window)
            
            # Compute rolling average inter-arrival time.
            if len(window) > 1:
                timestamps = [r[0] for r in window]
                diffs = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
                avg_interarrival = sum(diffs) / len(diffs)
            else:
                avg_interarrival = None
            
            # Query CMS for approximate frequency of current end_node.
            approx_freq = cms.query(end_node)
            
            print(f"Line: {line.strip()}")
            print(f"  -> Timestamp: {timestamp:.2f}")
            print(f"  -> Depth: {depth}")
            print(f"  -> End node: {end_node}")
            print(f"  -> Rolling average depth (last {len(window)} records): {avg_depth:.2f}")
            if avg_interarrival is not None:
                print(f"  -> Rolling average inter-arrival time: {avg_interarrival:.2f} s")
            else:
                print("  -> Not enough data for inter-arrival time yet.")
            print(f"  -> Approx frequency of end node '{end_node}' in window: {approx_freq}")
            print(f"  -> Label: {label}\n")

if __name__ == "__main__":

    process_dataset_cms_fixed_window('testtest..csv', window_size=10, epsilon=0.001, delta=0.01)
