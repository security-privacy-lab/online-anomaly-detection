import collections

def parse_wuil_line(line):
    """
    Parses a line from the WUIL file.
    """
    parts = line.strip().split('|')
    if len(parts) < 6:
        # Take care of invalid format of line
        return None  
    nodes_str = parts[5].strip()  
    nodes = [token for token in nodes_str.split('\\') if token]
    if not nodes:
        return None
    # Depth is the number of nodes
    return len(nodes)

def process_wuil_file(filename, window_size):
    """
    Processes the entire WUIL file.
    """
    window = collections.deque(maxlen=window_size)
    with open(filename, 'r') as f:
        for line in f:
            depth = parse_wuil_line(line)
            if depth is None:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            window.append(depth)
            avg = sum(window) / len(window)
            print(f"Line: {line.strip()}")
            print(f"  -> Current depth: {depth}, Rolling average (last {len(window)}): {avg}\n")

# Example usage:
if __name__ == "__main__":
    # Replace 'wuil_data.txt' with your actual file path.
    process_wuil_file('user1_log.txt', window_size=20)
