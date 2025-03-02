
import collections
from datetime import datetime

def parse_wuil_line(line):
    """
    Parses a line from the WUIL file.
    
    Expected format:
      index|date|time|id|someField|nodes
    """
    parts = line.strip().split('|')
    if len(parts) < 6:
        return None  # invalid format
    
    # Parse date and time.
    date_str = parts[1].strip()  # e.g., "15/04/2012"
    time_raw = parts[2].strip()  # e.g., "13:46:06 p.m."
    time_str = time_raw.split()[0]  # "13:46:06" (assuming 24-hour format)
    dt_str = f"{date_str} {time_str}"
    try:
        dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
    except ValueError as e:
        print(f"Error parsing datetime from '{dt_str}': {e}")
        return None
    timestamp = dt.timestamp()
    
    # Parse nodes (chain).
    nodes_str = parts[5].strip()  # e.g., "0\\1\\2\\3\\4"
    nodes = [token for token in nodes_str.split('\\') if token]
    if not nodes:
        return None
    depth = len(nodes)
    end_node = nodes[-1]
    return timestamp, depth, end_node

def process_wuil_file_window(filename, window_size=10):
    """
    Processes the entire WUIL file using a fixed window of size 'window_size'.
    
    For each record, it:
      - Parses the timestamp, depth (chain length), and end node.
      - Maintains a deque of the last 'window_size' records.
      - Computes the average depth over the window.
      - Computes the average inter-arrival time (time differences between consecutive timestamps).
      - Computes the frequency of the current end node in the window.
    """
    window = collections.deque(maxlen=window_size)
    
    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_wuil_line(line)
            if parsed is None:
                print(f"Skipping invalid line: {line.strip()}")
                continue
            timestamp, depth, end_node = parsed
            
            # Append the current record as a tuple: (timestamp, depth, end_node)
            window.append((timestamp, depth, end_node))
            
            # Compute the average depth over the window.
            avg_depth = sum(record[1] for record in window) / len(window)
            
            # Compute the average inter-arrival time using timestamps.
            if len(window) > 1:
                timestamps = [record[0] for record in window]
                # Compute differences between consecutive timestamps.
                diffs = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
                avg_interarrival = sum(diffs) / len(diffs)
            else:
                avg_interarrival = None
            
            # Compute frequency of end nodes in the window.
            end_nodes = [record[2] for record in window]
            freq_dict = collections.Counter(end_nodes)
            current_end_freq = freq_dict[end_node]
            
            print(f"Line: {line.strip()}")
            print(f"  -> Timestamp: {timestamp:.2f}")
            print(f"  -> Depth (chain length): {depth}")
            print(f"  -> End node: {end_node}")
            print(f"  -> Rolling average depth (last {len(window)} records): {avg_depth:.2f}")
            if avg_interarrival is not None:
                print(f"  -> Rolling average inter-arrival time: {avg_interarrival:.2f} seconds")
            else:
                print("  -> Not enough data for inter-arrival time yet.")
            print(f"  -> Frequency of end node '{end_node}' in window: {current_end_freq}\n")

if __name__ == "__main__":
    # Replace 'user1_log.txt' with your actual file path.
    process_wuil_file_window('user1_log.txt', window_size=10)
