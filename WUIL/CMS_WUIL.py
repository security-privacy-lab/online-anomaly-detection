import math, random, collections
from datetime import datetime

class CountMinSketch:
    def __init__(self, epsilon, delta):
        # How high accuracy should be?
        self.epsilon = epsilon
        # How confident we want to be in that accuracy.
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
# Parsing function for the dataset.
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
    
    # Return all separate pieces:
    # timestamp, source depth, destination depth, source end, destination end, label.
    return timestamp, src_depth, dst_depth, src_end, dst_end, label

#########################################
# Process dataset in groups of 10 records and print a summary.
#########################################

def process_dataset_in_groups(filename, group_size=10, epsilon=0.001, delta=0.01):
    group_records = []
    group_number = 1

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_dataset_line(line)
            if parsed is None:
                continue
            group_records.append(parsed)
            
            if len(group_records) == group_size:
                # Unpack the group data
                timestamps = [rec[0] for rec in group_records]
                src_depths = [rec[1] for rec in group_records]
                dst_depths = [rec[2] for rec in group_records]
                
                avg_src_depth = sum(src_depths) / group_size
                avg_dst_depth = sum(dst_depths) / group_size
                if len(timestamps) > 1:
                    diffs = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
                    avg_interarrival = sum(diffs) / len(diffs)
                else:
                    avg_interarrival = 0.0

                # Use Count-Min Sketch to get approximate frequency counts for each depth.
                cms_source = CountMinSketch(epsilon, delta)
                cms_destination = CountMinSketch(epsilon, delta)
                for rec in group_records:
                    # Update CMS with the source and destination depth values.
                    cms_source.update(rec[1], 1)
                    cms_destination.update(rec[2], 1)
                
                # Get frequency counts for each unique depth in the group.
                unique_src_depths = set(src_depths)
                unique_dst_depths = set(dst_depths)
                freq_src = {depth: cms_source.query(depth) for depth in unique_src_depths}
                freq_dst = {depth: cms_destination.query(depth) for depth in unique_dst_depths}
                
                print(f"Group {group_number} (records {group_number * group_size - group_size + 1} to {group_number * group_size}):")
                print(f"  Average Source Depth: {avg_src_depth:.2f}")
                print(f"  Average Destination Depth: {avg_dst_depth:.2f}")
                print(f"  Average Time Gap: {avg_interarrival:.2f} s")
                print(f"  Approx Frequency of Source Depths: {freq_src}")
                print(f"  Approx Frequency of Destination Depths: {freq_dst}\n")
                
                # Reset for the next group.
                group_records = []
                group_number += 1

if __name__ == "__main__":
    process_dataset_in_groups('testtest..csv', group_size=10, epsilon=0.001, delta=0.01)
