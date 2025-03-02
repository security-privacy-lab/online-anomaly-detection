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

def parse_wuil_line(line):
    parts = line.strip().split('|')
    if len(parts) < 6:
        return None
    date_str = parts[1].strip()    
    time_raw = parts[2].strip()    
    time_str = time_raw.split()[0]  
    dt_str = f"{date_str} {time_str}"
    try:
        dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
    except ValueError as e:
        print(f"Error parsing datetime from '{dt_str}': {e}")
        return None
    timestamp = dt.timestamp()
    nodes_str = parts[5].strip()   # e.g., "0\\1\\2\\3\\4"
    nodes = [token for token in nodes_str.split('\\') if token]
    if not nodes:
        return None
    depth = len(nodes)
    end_node = nodes[-1]
    return timestamp, depth, end_node



def process_wuil_file_cms_fixed_window(filename, window_size=10, epsilon=0.001, delta=0.01):

    cms = CountMinSketch(epsilon, delta)
    
    window = collections.deque()
    rolling_depth_sum = 0.0

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_wuil_line(line)
            if parsed is None:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            timestamp, depth, end_node = parsed
            
            
            if len(window) == window_size:
                oldest_timestamp, oldest_depth, oldest_end_node = window.popleft()
                cms.update(oldest_end_node, -1)
                rolling_depth_sum -= oldest_depth
            
            
            window.append((timestamp, depth, end_node))
            cms.update(end_node, 1)
            rolling_depth_sum += depth
            
            
            avg_depth = rolling_depth_sum / len(window)
            
            
            if len(window) > 1:
                timestamps = [record[0] for record in window]
                
                diffs = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
                avg_interarrival = sum(diffs) / len(diffs)
            else:
                avg_interarrival = None
            
           
            approx_freq = cms.query(end_node)
            
            print(f"Line: {line.strip()}")
            print(f"  -> Timestamp: {timestamp:.2f}")
            print(f"  -> Depth (chain length): {depth}")
            print(f"  -> End node: {end_node}")
            print(f"  -> Rolling average depth (last {len(window)} records): {avg_depth:.2f}")
            if avg_interarrival is not None:
                print(f"  -> Rolling avg inter-arrival time: {avg_interarrival:.2f} s")
            else:
                print("  -> Not enough data for inter-arrival time yet.")
            print(f"  -> Approx frequency of end node '{end_node}' in window: {approx_freq}\n")

if __name__ == "__main__":
    
    process_wuil_file_cms_fixed_window('user1_log.txt', window_size=10, epsilon=0.001, delta=0.01)
