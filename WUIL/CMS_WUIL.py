import math, random
import collections

class CountMinSketch:
    """
    A simple Count-Min Sketch implementation.
    """
    def __init__(self, epsilon, delta):
        # Width and depth based on error and confidence parameters.
        self.epsilon = epsilon
        self.delta = delta
        self.width = int(math.ceil(math.e / epsilon))
        self.depth = int(math.ceil(math.log(1 / delta)))
        self.table = [[0] * self.width for _ in range(self.depth)]
        # Random seeds for each hash function.
        self.hash_seeds = [random.randint(1, 100000) for _ in range(self.depth)]
    
    def _hash(self, item, i):
        # Convert item to string to ensure consistency.
        return (hash(str(item)) + self.hash_seeds[i]) % self.width
    
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
    """
    Parses a line from the WUIL file.
    
    Returns a tuple (depth, end_node), where:
      - depth is the number of nodes in the chain.
      - end_node is the last node in the chain.
    """
    parts = line.strip().split('|')
    if len(parts) < 6:
        return None  # invalid format
    nodes_str = parts[5].strip()
    # Split on backslash.
    nodes = [token for token in nodes_str.split('\\') if token]
    if not nodes:
        return None
    depth = len(nodes)
    end_node = nodes[-1]
    return depth, end_node

def process_wuil_file_cms(filename, epsilon=0.001, delta=0.01):
    """
    Processes the entire WUIL file using CMS.
    
    For each record:
      - Parses the depth (chain length) and the end node.
      - Updates two CMS structures for depth (one for counts, one for weighted sum)
        so that an overall average depth can be approximated.
      - Updates a CMS for the end node counts.
    
    Then, it reports:
      - The approximate average depth so far.
      - The approximate frequency for the current record's end node.
    
    Parameters:
      - filename: Path to the file.
      - epsilon, delta: CMS parameters (error and confidence).
    """
    cms_count = CountMinSketch(epsilon, delta)
    cms_sum = CountMinSketch(epsilon, delta)
    cms_end = CountMinSketch(epsilon, delta)
    
    observed_depths = set()
    observed_end_nodes = set()
    
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_wuil_line(line)
            if parsed is None:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            depth, end_node = parsed
            
            # Update CMS for depths.
            cms_count.update(depth, 1)
            cms_sum.update(depth, depth)
            observed_depths.add(depth)
            
            # Update CMS for end nodes.
            cms_end.update(end_node, 1)
            observed_end_nodes.add(end_node)
            
            # Approximate overall average depth so far:
            total_count = sum(cms_count.query(d) for d in observed_depths)
            total_sum = sum(cms_sum.query(d) for d in observed_depths)
            avg_depth = total_sum / total_count if total_count > 0 else 0
            
            # For the current record's end node, get approximate frequency.
            end_node_freq = cms_end.query(end_node)
            
            print(f"Line: {line.strip()}")
            print(f"  -> Observed depth (chain length): {depth}")
            print(f"  -> CMS-based approximate average depth so far: {avg_depth:.2f}")
            print(f"  -> Current end node: {end_node}, Approx. frequency: {end_node_freq}\n")

# Example usage:
if __name__ == "__main__":
    # Replace 'user1_log.txt' with your actual file path.
    process_wuil_file_cms('user1_log.txt', epsilon=0.001, delta=0.01)
