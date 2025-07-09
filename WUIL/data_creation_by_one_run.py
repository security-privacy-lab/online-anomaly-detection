import pandas as pd
import random
import os

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

columns = ['ID', 'Date', 'Time', 'Session_ID', 'Depth', 'Path', 'Label']


# Load the dataset
def load_dataset(filename): 
    try:
        dataset_loaded = pd.read_csv(f"{filename}.txt", sep='|', header=None, names = columns)
        print(f"{GREEN}Successfully loaded file | Name : {filename}.txt | Size : {len(dataset_loaded)}{RESET}")
        return dataset_loaded
    except FileNotFoundError:
        raise FileNotFoundError("The file has not been found. Please try again.")

# Preprocess the malicious dataset
def preprocess_dataset(dataset, benign): 
    dataset = dataset.sort_values(by='Session_ID').reset_index(drop=True)
    if benign:
        dataset['Label'] = 0
    else:
        dataset['Label'] = 1
    return dataset

def five_minute_injection(benign_dataset, malicious_dataset):
    """
    Injects malicious logs into 5-minute gaps in session IDs and prints the starting indexes neatly.
    The available gap indexes (where a gap exists) are divided into four quartile groups based on their count.
    When a gap is selected, its relative position (as a percentage of the dataset) is calculated and shown.
    """
    
    # Ensure malicious_dataset is a list.
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]

    injected_dataset = benign_dataset.copy()

    # Calculate time differences between consecutive Session_IDs (assumed to be integers)
    diff = injected_dataset['Session_ID'].astype(int).diff(-1).abs()
    gap_indexes = diff[diff >= 301].index.tolist()

    if not gap_indexes:
        raise Exception("No suitable 5-minute gaps found. Exiting.")

    # Helper function to divide gap indexes into quartile groups based on their count.
    def build_quartile_groups(gap_idxs):
        gap_idxs_sorted = sorted(gap_idxs)
        n = len(gap_idxs_sorted)
        quartile_groups = {}
        # Calculate indices for quartile splits.
        q1 = n // 4
        q2 = n // 2
        q3 = (3 * n) // 4

        quartile_groups["0-25%"] = gap_idxs_sorted[:q1]
        quartile_groups["25-50%"] = gap_idxs_sorted[q1:q2]
        quartile_groups["50-75%"] = gap_idxs_sorted[q2:q3]
        quartile_groups["75-100%"] = gap_idxs_sorted[q3:]
        return quartile_groups

    quartile_groups = build_quartile_groups(gap_indexes)

    while malicious_dataset:
        try:
            print("\nAvailable 5-minute gap indexes divided into quartile groups:")
            for quartile, indexes in quartile_groups.items():
                if indexes:
                    print(f"  {quartile} block: " + ", ".join(str(idx) for idx in indexes))
            
            # Ask the user to choose a gap index.
            gap_index = int(input("Enter the index of the gap to inject malicious logs: "))
            
            if gap_index not in gap_indexes:
                print("Error: Invalid choice! Please enter a valid index from the available quartile blocks.")
                continue

            # Compute the relative percentage of the dataset for the chosen index.
            total_length = len(injected_dataset)
            relative_percent = (gap_index / total_length) * 100

            # Determine which quartile group the chosen gap_index belongs to.
            chosen_quartile = None
            for quartile, indexes in quartile_groups.items():
                if gap_index in indexes:
                    chosen_quartile = quartile
                    break

            # Provide additional context within the chosen quartile.
            group = quartile_groups.get(chosen_quartile, [])
            pos = group.index(gap_index)
            prev_idx = group[pos-1] if pos > 0 else None
            next_idx = group[pos+1] if pos < len(group)-1 else None

            print(f"\nYou have chosen gap index {gap_index}, which is in the {chosen_quartile} block.")
            print(f"This corresponds to approximately {relative_percent:.1f}% of the dataset.")
            if prev_idx is not None:
                print(f"  Previous gap in this block: {prev_idx}")
            if next_idx is not None:
                print(f"  Next gap in this block: {next_idx}")

            # Pop the first malicious file and adjust its Session_ID values.
            malicious_file = malicious_dataset.pop(0)
            malicious_file["Session_ID"] = (
                malicious_file["Session_ID"].astype(int)
                - int(malicious_file.loc[0, "Session_ID"])
                + 1
                + int(injected_dataset.loc[gap_index, "Session_ID"])
            )

            # Inject the malicious logs into the dataset.
            injected_dataset = pd.concat([
                injected_dataset.iloc[:gap_index+1],
                malicious_file,
                injected_dataset.iloc[gap_index+1:]
            ]).reset_index(drop=True)

            # Update gap_indexes to account for the inserted malicious logs.
            new_gap_indexes = []
            for idx in gap_indexes:
                if idx == gap_index:
                    continue  # used gap index
                elif idx > gap_index:
                    new_gap_indexes.append(idx + len(malicious_file))
                else:
                    new_gap_indexes.append(idx)
            gap_indexes = new_gap_indexes

            # Rebuild the quartile groups based on the updated gap_indexes.
            quartile_groups = build_quartile_groups(gap_indexes)

        except ValueError:
            print("Error: Invalid input! Please enter a valid integer index.")

    print("Malicious logs successfully injected.")
    return injected_dataset

def five_minute_injection_by_percentile(benign_dataset, malicious_dataset):
    """
    Injects malicious logs into 5-minute gaps in session IDs and prints the starting indexes neatly.
    The available gap indexes (where a gap exists) are divided into four quartile groups based on their count.
    When a gap is selected, its relative position (as a percentage of the dataset) is calculated and shown.
    """
    
    # Ensure malicious_dataset is a list.
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]

    injected_dataset = benign_dataset.copy()

    # Calculate time differences between consecutive Session_IDs (assumed to be integers)
    diff = injected_dataset['Session_ID'].astype(int).diff(-1).abs()
    gap_indexes = diff[diff >= 301].index.tolist()
    gap_indexes_modified = gap_indexes.copy()

    if not gap_indexes:
        raise Exception("No suitable 5-minute gaps found. Exiting.")

    while malicious_dataset:
        try:
            # Ask the user to choose a gap index.
            insert_loc = int(input("Enter a percentage (0-100) to specify where to inject malicious data (e.g., 50% = middle): "))
            
            if insert_loc < 0 or insert_loc > 100:
                print("Error: Invalid choice! Please enter a valid percentage (0-100).")
                continue

            # Compute the relative percentage of the dataset for the chosen index.
            total_length = len(injected_dataset)
            input_index = int((insert_loc/100)*total_length)

            def bin_search(i, j, target):
                while i < j-1:
                    mid = (i+j)//2

                    if gap_indexes[mid] == target:
                        return mid
                    
                    if gap_indexes[mid] > target:
                        j = mid
                    else:
                        i = mid

                return i if target - gap_indexes[i] < gap_indexes[j] - target else j

            insert_index = gap_indexes_modified[bin_search(0, len(gap_indexes)-1, input_index)]

            # Pop the first malicious file and adjust its Session_ID values.
            malicious_file = malicious_dataset.pop(0)
            malicious_file["Session_ID"] = (
                malicious_file["Session_ID"].astype(int)
                - int(malicious_file.loc[0, "Session_ID"])
                + 1
                + int(injected_dataset.loc[insert_index, "Session_ID"])
            )

            # Inject the malicious logs into the dataset.
            injected_dataset = pd.concat([
                injected_dataset.iloc[:insert_index+1],
                malicious_file,
                injected_dataset.iloc[insert_index+1:]
            ]).reset_index(drop=True)

            # Update gap_indexes_modified to account for the inserted malicious logs.
            new_gap_indexes = []
            for idx in gap_indexes:
                if idx > insert_index:
                    new_gap_indexes.append(idx + len(malicious_file))
                else:
                    new_gap_indexes.append(idx)
            gap_indexes_modified = new_gap_indexes

        except ValueError:
            print("Error: Invalid input! Please enter a valid integer index.")

    print("Malicious logs successfully injected.")
    return injected_dataset
def five_minute_injection_continuous(benign_dataset, malicious_dataset):
    """
    Like five_minute_injection, but overrides the malicious rows'
    Session_ID to be continuous integers (no jumps or backward moves).
    """
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]
    injected_dataset = benign_dataset.copy()

    # find gaps of ≥301 in Session_ID
    diff = injected_dataset['Session_ID'].astype(int).diff(-1).abs()
    gap_indexes = diff[diff >= 301].index.tolist()
    if not gap_indexes:
        raise Exception("No suitable 5-minute gaps found.")

    # simple quartile grouping reuse if you like, or just show the raw list:
    while malicious_dataset:
        print(f"Available gap indexes: {gap_indexes}")
        idx = int(input("Choose gap index to inject at: "))
        if idx not in gap_indexes:
            print("Invalid index, try again."); continue

        # pop one malicious chunk
        mal = malicious_dataset.pop(0).copy()
        start_ts = int(injected_dataset.loc[idx, "Session_ID"])
        # override to be strictly consecutive
        mal["Session_ID"] = list(range(start_ts + 1,
                                        start_ts + 1 + len(mal)))

        # inject
        injected_dataset = pd.concat([
            injected_dataset.iloc[:idx+1],
            mal,
            injected_dataset.iloc[idx+1:]
        ]).reset_index(drop=True)

        # shift remaining gap indexes forward by len(mal)
        gap_indexes = [
            i + len(mal) if i > idx else i
            for i in gap_indexes
            if i != idx
        ]

    print("Injection complete (continuous timestamps).")
    return injected_dataset

def random_injection(benign_dataset, malicious_dataset):
    """Injects malicious logs randomly into the dataset."""

    # Correct input to type list
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]

    injected_dataset = benign_dataset.copy()
    
    for mal_dataset in malicious_dataset:
        injected_indices = []
        
        for _, malicious_row in mal_dataset.iterrows():
            malicious_data = {
                'Session_ID': malicious_row['Session_ID'],
                'Depth': malicious_row['Depth'],
                'Path': malicious_row['Path'],
                'Label': malicious_row['Label']
            }
            malicious_row_df = pd.DataFrame([malicious_data])

            random_index = random.randint(0, len(injected_dataset)) 
            injected_indices.append(random_index)
            injected_dataset = pd.concat([
                injected_dataset.iloc[:random_index],  
                malicious_row_df,                      
                injected_dataset.iloc[random_index:]  
            ]).reset_index(drop=True)
        
    print("{GREEN}Malicious logs successfully injected at random indices.{RESET}")
    return injected_dataset

def organized_injection(benign_dataset, malicious_dataset):
    """Organizes malicious logs into the dataset in order."""

    # Correct input to type list
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]

    injected_dataset = benign_dataset.copy()
    
    for mal_data in malicious_dataset:
        injected_dataset = pd.concat((injected_dataset, mal_data), ignore_index=True)
        injected_dataset = injected_dataset.sort_values(by="Session_ID", ascending=True)
    return injected_dataset

def customize_saving_method(edges_df, filepath):
    """Allows the user to customize saving columns."""
    new_columns = []
    
    while True:
        question = input(f"What do you want to input? Available columns: {CYAN}{list(edges_df.columns)}{RESET}\nType 'q' to quit: ")
        if question == 'q':
            break
        elif question in edges_df.columns:
            new_columns.append(question)
            print(f'Current selected columns: {CYAN}{new_columns}{RESET}')
        else:
            print(f"{YELLOW}Error: Invalid column. Please try again.{YELLOW}")
    
    if new_columns:
        separator_type = input("What is your preferred separator?")
        file_type = input("What do you want your file formatting to be?")
        edges_df[new_columns].to_csv(f"{filepath}.{file_type}", sep=f'{separator_type}', index=False)
        print(f"{GREEN}Successfully created the dataset! Exiting...{RESET}")
    else:
        print(f"{YELLOW}No columns selected. Exiting...{RESET}")

def save_as_MAD(injected_dataset, filepath):
    edges = []
    prev_path = None
    min_timestamp = injected_dataset['Session_ID'].min()
    node_mapping = {}
    next_id = 0 

    # Build the edges dataset
    for _, row in injected_dataset.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']
        norm_timestamp = int(timestamp) - int(min_timestamp) + 1  # Normalize timestamp

        # Assign unique integer IDs to paths
        if current_path not in node_mapping:
            node_mapping[current_path] = next_id
            next_id += 1
        current_path_mapped = node_mapping[current_path]

        # Handle self-loops and sequential edges
        if prev_path == current_path:
            edges.append({
                'src_node': current_path_mapped,
                'dst_node': current_path_mapped,
                'timestamp': norm_timestamp,
                'label': label
            })
        elif prev_path is not None:
            if prev_path not in node_mapping:
                node_mapping[prev_path] = next_id
                next_id += 1
            prev_path_mapped = node_mapping[prev_path]
            edges.append({
                'src_node': prev_path_mapped,
                'dst_node': current_path_mapped,
                'timestamp': norm_timestamp,
                'label': label
            })
        
        prev_path = current_path

    # Convert edges to DataFrame
    edges_df = pd.DataFrame(edges)
    edges_df[['src_node', 'dst_node', 'timestamp']] = edges_df[['src_node', 'dst_node', 'timestamp']].astype(int)
    edges_df[['src_node', 'dst_node', 'timestamp']].to_csv(
        f'{filepath}_data.txt', sep=',', header=False, index=False
    )

    # Filter for malicious edges (where label == 1)
    dataset_filtered = edges_df[edges_df['label'] == 1]

    # Create query dataset
    query = dataset_filtered[['src_node', 'dst_node']].drop_duplicates()
    query = query[['src_node', 'dst_node']]
    query = query.sort_values(by=['src_node','dst_node'])
    query.to_csv(f'{filepath}_queries.txt', sep=',', header=False, index=False)

    # Work on ground truth values
    query_edge = set(query.apply(tuple, axis=1))
    related_edges = edges_df[edges_df[['src_node', 'dst_node']].apply(tuple, axis=1).isin(query_edge)]
    related_edges = related_edges.sort_values(by=['src_node', 'dst_node', 'timestamp'])
    related_edges.to_csv(f"{filepath}_gt.txt",sep=',',index=False, header=None)

    print("Processing complete. Data saved to:", filepath)

def save_as_anomrank_or_f_fade(injected_dataset, filepath):
    edges = []
    prev_path = None
    node_mapping = {}  # Store unique path-to-integer mappings
    next_id = 0  # Start node IDs from 0

    # Calculate the minimum timestamp in the dataset
    min_timestamp = injected_dataset['Session_ID'].min()

    for _, row in injected_dataset.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']

        # Normalize the timestamp: subtract the minimum and add 1 so the smallest becomes 1
        norm_timestamp = int(timestamp) - int(min_timestamp) + 1

        # Map the current path to a unique integer
        if current_path not in node_mapping:
            node_mapping[current_path] = next_id
            next_id += 1
        current_path_mapped = node_mapping[current_path]

        # Create an edge entry. If the current path is the same as the previous, make it a self-edge.
        if prev_path == current_path:
            edges.append({
                'src_node': current_path_mapped,
                'dst_node': current_path_mapped,
                'timestamp': norm_timestamp,
                'label': label
            })
        elif prev_path is not None:
            # Map the previous path if it isn't mapped yet
            if prev_path not in node_mapping:
                node_mapping[prev_path] = next_id
                next_id += 1
            prev_path_mapped = node_mapping[prev_path]
            edges.append({
                'src_node': prev_path_mapped,
                'dst_node': current_path_mapped,
                'timestamp': norm_timestamp,
                'label': label
            })
        
        prev_path = current_path

    # Convert to DataFrame
    edges_df = pd.DataFrame(edges)

    # Save dataset in the required format: timestamp, src_node, dst_node, label.
    edges_df[[ 'timestamp','src_node', 'dst_node','label']].to_csv(
        f'{filepath}.txt', sep=' ', header=False, index=False
    )


def save_as_sedanspot(edges_df, filepath):
    """Saves the dataset in SedanSpot format."""
    edges_df[['timestamp', 'src_node', 'dst_node', 'weight', 'label']].to_csv(
        f'{filepath}.csv', sep=',', header=False, index=False
    )

def save_as_midas(edges_df, filepath):
    """Saves the dataset in Midas format."""

    combined = pd.concat([edges_df["src_node"], edges_df["dst_node"]])
    factor_mapping = {val: idx for (idx, val) in enumerate(combined.unique())}

    edges_df["src_node"] = edges_df["src_node"].map(factor_mapping)
    edges_df["dst_node"] = edges_df["dst_node"].map(factor_mapping)

    edges_df[['src_node', 'dst_node', 'timestamp']].to_csv(
        f'{filepath}_features.csv', sep=',', header=False, index=False
    )
    edges_df[['label']].to_csv(
        f'{filepath}_labels.csv', sep=',', header=False, index=False
    )

    with open(f"{filepath}_shape.txt", "w") as f:
        f.write(f"{edges_df.shape[0]}")

def run():

    # Step 1: load benign dataset
    loading_file = True
    while loading_file:
        filename = input("Enter the benign dataset file name (without extension): ")

        try:
            benign_dataset = load_dataset(filename)
            benign_dataset = preprocess_dataset(benign_dataset, benign=True)
            loading_file = False
        except Exception as e:
            print(f"{YELLOW}Error: {e}{RESET}")

    
    injected_dataset = pd.DataFrame()
    awaiting_response = True
    # Make choices
    while awaiting_response:
        choice = input(f"Choose injection method: {CYAN}rj{RESET} (random), {CYAN}5m{RESET} (5-minute batch), {CYAN}5mc{RESET} (5-minute continuous), or {CYAN}or{RESET} (organized): ")

        
        if choice in ['rj', '5m', 'or', '5mc']:
            while True:
                try:
                    num_logs = int(input("How many malicious files do you want to inject? : "))
                    break
                except ValueError as e:
                    print(f"{YELLOW} Error: Please enter an integer.{RESET}")
            
            malicious_datasets = []

            while len(malicious_datasets) < num_logs:
                filename = input(f"Enter the malicious dataset file name (without extension): ")
                try:
                    malicious_dataset = load_dataset(filename)
                    malicious_dataset = preprocess_dataset(malicious_dataset, benign=False)
                    malicious_datasets.append(malicious_dataset)
                except Exception as e:
                    print(f"{YELLOW}Error: {e}{RESET}")
            
            
            if choice == 'rj':
                injected_dataset = random_injection(benign_dataset, malicious_datasets)
            elif choice == '5m':
                injected_dataset = five_minute_injection(benign_dataset, malicious_datasets)
            elif choice == 'or':
                injected_dataset = organized_injection(benign_dataset, malicious_datasets)
            elif choice == '5mc':
                injected_dataset = five_minute_injection_continuous(benign_dataset, malicious_datasets)

            print(f"{GREEN}Original Dataset Length: {len(benign_dataset)}{RESET}")
            print(f'{GREEN}Injected Dataset Length: {len(injected_dataset)}{RESET}')
            awaiting_response = False
        else: 
            print(f"{YELLOW}Error: Invalid choice! Please try again.{YELLOW}")


    # Make self-edging graph here
    edges = []
    prev_path = None
    for _, row in injected_dataset.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']

        if prev_path == current_path:
            edges.append({'src_node': current_path, 'dst_node': current_path, 'timestamp': timestamp, 'weight': 1, 'label': label})
        elif prev_path is not None:
            edges.append({'src_node': prev_path, 'dst_node': current_path, 'timestamp': timestamp, 'weight': 1, 'label': label})
        
        prev_path = current_path
        
    print(f"""
    Choose file save format:
    1. {CYAN}Anomrank/F-Fade{RESET}
    2. {CYAN}Sedanspot{RESET}
    3. {CYAN}MAD{RESET}
    4. {CYAN}MIDAS{RESET}
    5. Customize
    """)


    edges_df = pd.DataFrame(edges)

    awaiting_response = True

    os.makedirs("./custom_data", exist_ok=True)

    while awaiting_response:
        file_type = input(f"Select format ({CYAN}1{RESET}, {CYAN}2{RESET}, {CYAN}3{RESET} or {CYAN}4{RESET}): ")
        filename = input("Enter the desired file name: ")
        filepath = os.path.join("./custom_data", filename)
        if file_type == '1':
            save_as_anomrank_or_f_fade(injected_dataset, filepath)
            awaiting_response = False
        elif file_type == '2':
            save_as_sedanspot(edges_df, filepath)
            awaiting_response = False
        elif file_type == '3':
            save_as_MAD(injected_dataset, filepath)
            awaiting_response = False
        elif file_type == '4':
            save_as_midas(edges_df, filepath)
            awaiting_response = False
        elif file_type == '5':
            customize_saving_method(edges_df, filepath)
            awaiting_response = False
        else:
            print(f"{YELLOW}Error: Invalid choice! Please try again.{RESET}")

    print(f"{GREEN}Successfully saved data in custom_data folder: exiting...{RESET}")
if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print(f"{YELLOW}\nKeyboard interrupt: exiting...{RESET}")
