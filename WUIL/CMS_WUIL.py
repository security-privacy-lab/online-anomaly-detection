import math, random, collections
from datetime import datetime

class CountMinSketch:
    def __init__(self, epsilon, delta):
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

#########################################
# Updated parsing function for the dataset
#########################################

def parse_dataset_line(line):
    # Skip header lines or empty lines
    if line.lower().startswith("src_node") or not line.strip():
        return None
    parts = line.strip().split('|')
    if len(parts) < 4:
        return None
    src_chain = parts[0].strip()      
    dst_chain = parts[1].strip()      
    timestamp_str = parts[2].strip()  
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
    
    # Return all separate pieces: timestamp, source depth, destination depth, source end, destination end, label.
    return timestamp, src_depth, dst_depth, src_end, dst_end, label

#########################################
# Updated processing function to track source and destination depths separately
#########################################

def process_dataset_cms_fixed_window(filename, window_size=10, epsilon=0.001, delta=0.01):

    # Two separate Count-Min Sketches: one for source depth and one for destination depth.
    cms_source = CountMinSketch(epsilon, delta)
    cms_destination = CountMinSketch(epsilon, delta)
    
    # Window will store tuples: (timestamp, src_depth, dst_depth, src_end, dst_end, label)
    window = collections.deque()  
    rolling_src_depth_sum = 0.0
    rolling_dst_depth_sum = 0.0

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_dataset_line(line)
            if parsed is None:
                continue
            
            timestamp, src_depth, dst_depth, src_end, dst_end, label = parsed
            
            # If window is full, remove the oldest record.
            if len(window) == window_size:
                oldest = window.popleft()
                oldest_timestamp, oldest_src_depth, oldest_dst_depth, oldest_src_end, oldest_dst_end, oldest_label = oldest
                cms_source.update(oldest_src_depth, -1)
                cms_destination.update(oldest_dst_depth, -1)
                rolling_src_depth_sum -= oldest_src_depth
                rolling_dst_depth_sum -= oldest_dst_depth
            
            # Add new record.
            window.append((timestamp, src_depth, dst_depth, src_end, dst_end, label))
            cms_source.update(src_depth, 1)
            cms_destination.update(dst_depth, 1)
            rolling_src_depth_sum += src_depth
            rolling_dst_depth_sum += dst_depth
            
            # Compute rolling averages.
            avg_src_depth = rolling_src_depth_sum / len(window)
            avg_dst_depth = rolling_dst_depth_sum / len(window)
            
            # Compute rolling average inter-arrival time.
            if len(window) > 1:
                timestamps = [r[0] for r in window]
                diffs = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
                avg_interarrival = sum(diffs) / len(diffs)
            else:
                avg_interarrival = None
            
            # Query the CMS for the frequency of the current source and destination depths.
            approx_freq_source = cms_source.query(src_depth)
            approx_freq_destination = cms_destination.query(dst_depth)
            
            print(f"Line: {line.strip()}")
            print(f"  -> Timestamp: {timestamp:.2f}")
            print(f"  -> Source Depth: {src_depth}")
            print(f"  -> Destination Depth: {dst_depth}")
            print(f"  -> Average depth in source(last {len(window)} records): {avg_src_depth:.2f}")
            print(f"  -> Average depth in the destination (last {len(window)} records): {avg_dst_depth:.2f}")
            if avg_interarrival is not None:
                print(f"  -> Average time gap between each record in the window {avg_interarrival:.2f} s")
            else:
                print("  -> Not enough data for inter-arrival time yet.")
            print(f"  -> Approx frequency of Source Depth {src_depth} in window: {approx_freq_source}")
            print(f"  -> Approx frequency of Destination Depth {dst_depth} in window: {approx_freq_destination}")
            print(f"  -> Label: {label}\n")

if __name__ == "__main__":
    process_dataset_cms_fixed_window('testtest..csv', window_size=10, epsilon=0.001, delta=0.01)
