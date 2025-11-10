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

def save_as_anomrank_with_idle_padding(injected_dataset, filepath, bin_size=1, pad_label=0):
    import pandas as pd

    edges = []
    node_map, next_id = {}, 0

    def nid(p):
        nonlocal next_id
        if p not in node_map:
            node_map[p] = next_id
            next_id += 1
        return node_map[p]

    min_ts = int(injected_dataset['Session_ID'].min())
    prev_path = None
    prev_bin  = None

    for _, row in injected_dataset.iterrows():
        cur_path = row['Path']
        ts       = int(row['Session_ID'])
        label    = int(row['Label'])
        cur_id   = nid(cur_path)
        bin_t    = (ts - min_ts) // int(bin_size)

        if prev_path is None:
            prev_path = cur_path
            prev_bin  = bin_t
            continue

        # --- idle padding: fill missing bins with self-loop on prev node ---
        if bin_t - prev_bin > 1:
            prev_id = nid(prev_path)
            for b in range(prev_bin + 1, bin_t):
                edges.append({
                    'time_bin': b,
                    'src_id'  : prev_id,
                    'dst_id'  : prev_id,
                    'label'   : int(pad_label)
                })

        # 실제 전이(edge)
        prev_id = nid(prev_path)
        edges.append({
            'time_bin': bin_t,
            'src_id'  : prev_id,
            'dst_id'  : cur_id,
            'label'   : label
        })

        prev_path = cur_path
        prev_bin  = bin_t

    if not edges:
        print("No edges to save (idle padding).")
        return

    df = pd.DataFrame(edges, columns=['time_bin','src_id','dst_id','label']).astype(int)
    df = df.sort_values(['time_bin','src_id','dst_id'])
    df[['time_bin','src_id','dst_id','label']].to_csv(f'{filepath}.txt', sep=' ', header=False, index=False)
    print(f"[AnomRank idle-padded] saved → {filepath}.txt")

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

def five_minute_injection_continuous_improved(benign_dataset, malicious_dataset, enforce_shift=True):
    """
    Insert malicious chunks into 5-minute gaps using continuous integer Session_IDs:
      - Each malicious chunk's Session_IDs become consecutive integers starting at gap_session_id + 1.
      - If the malicious chunk would overlap or exceed the following benign timestamp,
        and enforce_shift is True, all subsequent benign timestamps are shifted forward
        to avoid overlap.
      - Supports multiple malicious chunks (list or single DataFrame).
    Returns the injected dataset (reset index).
    """
    # Normalize malicious_dataset to a list of DataFrames
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset.copy()]
    else:
        malicious_dataset = [m.copy() for m in malicious_dataset]

    injected_dataset = benign_dataset.copy().reset_index(drop=True)

    # find gaps of >=301 between consecutive Session_ID
    diff = injected_dataset['Session_ID'].astype(int).diff(-1).abs()
    gap_indexes = diff[diff >= 301].index.tolist()
    if not gap_indexes:
        raise Exception("No suitable 5-minute gaps found.")

    def build_quartile_groups(gap_idxs):
        gap_idxs_sorted = sorted(gap_idxs)
        n = len(gap_idxs_sorted)
        q1 = n // 4
        q2 = n // 2
        q3 = (3 * n) // 4
        return {
            "0-25%": gap_idxs_sorted[:q1],
            "25-50%": gap_idxs_sorted[q1:q2],
            "50-75%": gap_idxs_sorted[q2:q3],
            "75-100%": gap_idxs_sorted[q3:]
        }

    quartile_groups = build_quartile_groups(gap_indexes)

    while malicious_dataset:
        print(f"\nAvailable gap indexes: {gap_indexes}")
        try:
            idx = int(input("Choose gap index to inject at: "))
        except ValueError:
            print("Invalid input; please enter an integer index.")
            continue

        if idx not in gap_indexes:
            print("Invalid index; choose from the available gap_indexes.")
            continue

        # Pop one malicious chunk and copy it
        mal = malicious_dataset.pop(0).copy().reset_index(drop=True)
        mal_len = len(mal)

        # Determine new consecutive timestamps starting at start_ts+1
        start_ts = int(injected_dataset.loc[idx, "Session_ID"])
        new_start = start_ts + 1
        new_mal_ts = list(range(new_start, new_start + mal_len))
        mal['Session_ID'] = new_mal_ts

        # Determine next existing timestamp (after insertion spot) if any
        next_existing_idx = idx + 1
        if next_existing_idx < len(injected_dataset):
            next_existing_ts = int(injected_dataset.loc[next_existing_idx, 'Session_ID'])
        else:
            next_existing_ts = None

        # If overlap would occur, compute shift needed
        shift_amount = 0
        if next_existing_ts is not None:
            last_mal_ts = new_mal_ts[-1]
            if last_mal_ts >= next_existing_ts:
                shift_amount = last_mal_ts - next_existing_ts + 1

        # Insert the malicious chunk
        injected_dataset = pd.concat([
            injected_dataset.iloc[:idx+1],
            mal,
            injected_dataset.iloc[idx+1:]
        ]).reset_index(drop=True)

        # If shift required and allowed, shift subsequent rows
        if shift_amount > 0 and enforce_shift:
            # The start index of rows to shift is idx + len(mal) + 1
            shift_start = idx + len(mal) + 1
            if shift_start < len(injected_dataset):
                injected_dataset.loc[shift_start:, 'Session_ID'] = (
                    injected_dataset.loc[shift_start:, 'Session_ID'].astype(int) + shift_amount
                )
            print(f"Inserted {mal_len} rows at index {idx}; shifted subsequent rows by {shift_amount}.")

        else:
            print(f"Inserted {mal_len} rows at index {idx}; no shift needed.")

        # Safety pass: enforce strictly increasing Session_IDs (each >= prev + 1)
        sess = injected_dataset['Session_ID'].astype(int).to_numpy()
        for i in range(1, len(sess)):
            if sess[i] <= sess[i-1]:
                sess[i] = sess[i-1] + 1
        injected_dataset['Session_ID'] = sess

        # Recompute gap indexes and quartiles for further injections
        diff = injected_dataset['Session_ID'].astype(int).diff(-1).abs()
        gap_indexes = diff[diff >= 301].index.tolist()
        quartile_groups = build_quartile_groups(gap_indexes)

    print("Injection complete (continuous timestamps enforced).")
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

def organized_injection(benign_dataset, malicious_dataset, sort_by_session=True):
    """
    benign 뒤에 악성 청크들을 붙이고 (옵션) Session_ID로 정렬.
    - 모든 행에 source / attack_id 칼럼을 달아줌
    - 정렬하지 않으면 각 악성 청크는 연속 블록 → (start,end) 한 쌍으로 보고
    - 정렬하면 attack_id가 흩어질 수 있음 → 여러 (start,end) 구간으로 보고
    반환: injected_dataset, attack_spans
      attack_spans = { attack_id: [ (start_idx, end_idx), ... ] }  # 0-based, inclusive
    """
    import pandas as pd

    # 리스트로 통일
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]

    # 복사 및 초기 태깅
    injected = benign_dataset.copy().reset_index(drop=True)
    injected["source"] = "benign"
    injected["attack_id"] = 0   # benign은 0(비공격)

    attack_spans = {}
    next_attack_id = 1

    # 1) 정렬하지 않을 경우: 연속 블록 그대로 기록
    if not sort_by_session:
        for m in malicious_dataset:
            m = m.copy().reset_index(drop=True)
            m["Label"] = 1
            m["source"] = "malicious"
            m["attack_id"] = next_attack_id

            start = len(injected)              # 붙이기 전 시작 라인(0-based)
            injected = pd.concat([injected, m], ignore_index=True)
            end   = len(injected) - 1          # 붙인 뒤 종료 라인(포함)

            attack_spans[next_attack_id] = [(start, end)]
            next_attack_id += 1

        return injected, attack_spans

    # 2) 정렬하는 경우: 먼저 다 붙이고, 정렬 후 attack_id별로 끊어진 구간을 계산
    chunks = [injected]
    for m in malicious_dataset:
        m = m.copy().reset_index(drop=True)
        m["Label"] = 1
        m["source"] = "malicious"
        m["attack_id"] = next_attack_id
        chunks.append(m)
        next_attack_id += 1

    injected = pd.concat(chunks, ignore_index=True)

    # Session_ID 기준 정렬 → 행 인덱스가 새로 섞임
    injected = injected.sort_values(by="Session_ID", kind="mergesort").reset_index(drop=True)
    # (mergesort는 안정 정렬)

    # attack_id별로 연속 인덱스 구간을 계산
    for aid, g in injected.groupby("attack_id", sort=False):
        if aid == 0:
            continue  # benign은 스킵
        idxs = g.index.to_list()
        if not idxs:
            continue

        # 연속된 인덱스 조각으로 나눠 spans 생성
        spans = []
        s = e = idxs[0]
        for x in idxs[1:]:
            if x == e + 1:
                e = x
            else:
                spans.append((s, e))
                s = e = x
        spans.append((s, e))
        attack_spans[aid] = spans

    return injected, attack_spans


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
def save_as_SLADE(injected_dataset, dataset_name):
    os.makedirs("data", exist_ok=True)

    edges = []
    node_mapping = {}
    next_id = 1  # Start from 1 (0 is reserved for padding in SLADE/TGN)

    def get_node_id(path_str):
        nonlocal next_id
        if path_str not in node_mapping:
            node_mapping[path_str] = next_id
            next_id += 1
        return node_mapping[path_str]

    # Find the minimum timestamp for normalization
    min_ts = int(injected_dataset["Session_ID"].min())

    prev_path = None
    for _, row in injected_dataset.iterrows():
        cur_path = row["Path"]
        ts = int(row["Session_ID"])
        lab = int(row["Label"])

        # Normalize timestamp to start at 1
        norm_ts = ts - min_ts + 1

        cur_id = get_node_id(cur_path)

        if prev_path is None:
            prev_path = cur_path
            continue

        prev_id = get_node_id(prev_path)

        # Create an edge between previous and current path
        edges.append({
            "u": prev_id,
            "i": cur_id,
            "ts": norm_ts,
            "label": lab
        })

        prev_path = cur_path

    if not edges:
        print("No edges to save for SLADE.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(edges, columns=["u", "i", "ts", "label"]).astype(int)

    # Sort edges by timestamp to ensure chronological order
    df = df.sort_values("ts").reset_index(drop=True)

    # Assign unique edge indices after sorting
    df["idx"] = df.index.astype(int)

    # Save dataset in SLADE format
    out_path = f"data/ml_{dataset_name}.csv"
    df[["u", "i", "ts", "label", "idx"]].to_csv(out_path, index=False, header=True)
    print(f"{GREEN}SLADE file saved → {out_path}{RESET}")


def save_as_SLADE_tail_time_only(
    injected_dataset,
    dataset_name,
    pad_ticks=300,
    step=1,
    pad_label=0,
    fill_internal_gaps=True
):
    """
    Save in SLADE/TGN format with:
      - node IDs starting at 1,
      - timestamps normalized to start at 1,
      - optional internal idle filling per (u, i) pair,
      - tail padding (self-loops on the last node).
    Output columns: u, i, ts, label, idx
    """
    os.makedirs("data", exist_ok=True)

    edges = []
    node_mapping = {}
    next_id = 1  # Start node IDs from 1 (0 is reserved for padding in SLADE/TGN)

    def get_node_id(path_str):
        """Map each unique path to a unique integer node ID (starting from 1)."""
        nonlocal next_id
        if path_str not in node_mapping:
            node_mapping[path_str] = next_id
            next_id += 1
        return node_mapping[path_str]

    # Base timestamp normalization anchor
    min_ts = int(injected_dataset["Session_ID"].min())

    prev_path = None
    last_norm_ts = 0
    last_node_id = None

    # 1) Build base edges from the dataset (Path sequence -> edges)
    for _, row in injected_dataset.iterrows():
        cur_path = row["Path"]
        ts = int(row["Session_ID"])
        lab = int(row["Label"])
        norm_ts = ts - min_ts + 1  # normalize to start at 1

        cur_id = get_node_id(cur_path)

        if prev_path is None:
            # Skip the very first row (no previous node to connect from)
            prev_path = cur_path
            last_norm_ts = norm_ts
            last_node_id = cur_id
            continue

        prev_id = get_node_id(prev_path)

        edges.append({
            "u": prev_id,
            "i": cur_id,
            "ts": norm_ts,
            "label": lab
        })

        prev_path = cur_path
        last_norm_ts = norm_ts
        last_node_id = cur_id

    if not edges:
        print("No edges to save for SLADE (time-only).")
        return

    import pandas as pd
    df = pd.DataFrame(edges, columns=["u", "i", "ts", "label"]).astype(int)

    # 2) (Optional) Fill internal idle gaps per (u, i) pair
    if fill_internal_gaps:
        # Sort by pair then time to detect gaps
        df = df.sort_values(["u", "i", "ts"]).reset_index(drop=True)

        filler_rows = []
        # Iterate consecutive timestamps for each (u, i) group
        for (u, i), g in df.groupby(["u", "i"], sort=False):
            ts_vals = g["ts"].values
            # Walk through successive events for this pair
            for prev_ts, next_ts in zip(ts_vals[:-1], ts_vals[1:]):
                gap = next_ts - prev_ts
                if gap > step:
                    # Add idle edges at prev_ts + step, ..., next_ts - step
                    for t in range(prev_ts + step, next_ts, step):
                        filler_rows.append({"u": u, "i": i, "ts": t, "label": int(pad_label)})

        if filler_rows:
            df = pd.concat([df, pd.DataFrame(filler_rows)], ignore_index=True).astype(int)

    # 3) Tail padding (self-loops on the last node) to extend timeline
    if pad_ticks > 0 and last_node_id is not None:
        tail_rows = [{"u": last_node_id,
                      "i": last_node_id,
                      "ts": last_norm_ts + step * k,
                      "label": int(pad_label)}
                     for k in range(1, pad_ticks + 1)]
        df = pd.concat([df, pd.DataFrame(tail_rows)], ignore_index=True).astype(int)

    # 4) Final sort by time (global chronological order) and assign idx
    df = df.sort_values("ts").reset_index(drop=True)
    df["idx"] = df.index.astype(int)

    # 5) Save in SLADE format
    out_path = f"data/ml_{dataset_name}_timepad.csv"
    df[["u", "i", "ts", "label", "idx"]].to_csv(out_path, index=False, header=True)
    print(f"{GREEN}SLADE time-only file saved → {out_path}{RESET}")

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
                injected_dataset, attack_spans = organized_injection(benign_dataset, malicious_datasets)
                print(f"\n=== Injection Summary ===")
                for aid, spans in attack_spans.items():
                    for (s, e) in spans:
                        print(f"attack {aid}: rows {s}~{e} (count={e - s + 1})")

            elif choice == '5mc':
                injected_dataset = five_minute_injection_continuous_improved(benign_dataset, malicious_datasets)

            print(f"{GREEN}Original Dataset Length: {len(benign_dataset)}{RESET}")
            print(f'{GREEN}Injected Dataset Length: {len(injected_dataset)}{RESET}')
            awaiting_response = False
        else: 
            print(f"{YELLOW}Error: Invalid choice! Please try again.{YELLOW}")


    # Make self-edging graph here
    edges = []
    prev_path  = None
    prev_label = None

    for _, row in injected_dataset.iterrows():
        cur_path  = row['Path']
        timestamp = int(row['Session_ID'])
        cur_label = int(row['Label'])  # 현재 행의 라벨

        if prev_path is None:
            prev_path  = cur_path
            prev_label = cur_label
            continue

        # ✅ 엣지 라벨 규칙
        edge_label = prev_label        # 소스 기준 (가장 일반적)

        # ✅ 이 부분이 루프 내부에 들어가야 함
        if prev_path == cur_path:
            edges.append({
                'src_node': cur_path,
                'dst_node': cur_path,
                'timestamp': timestamp,
                'weight': 1,
                'label': edge_label
            })
        else:
            edges.append({
                'src_node': prev_path,
                'dst_node': cur_path,
                'timestamp': timestamp,
                'weight': 1,
                'label': edge_label
            })

        prev_path  = cur_path
        prev_label = cur_label

    edges_df = pd.DataFrame(edges)
        
    print(f"""
    Choose file save format:
    1. {CYAN}Anomrank/F-Fade{RESET}
    2. {CYAN}Sedanspot{RESET}
    3. {CYAN}MAD{RESET}
    4. {CYAN}MIDAS{RESET}
    5. {CYAN}Customize{RESET}
    6. {CYAN} SLADE{RESET}
    7. {CYAN} SLADE (time-only){RESET} (adds self-edges at the end with incrementing timestamps)
    8. {CYAN} AnomRank with idle padding{RESET} (adds self-edges during idle periods)
    """)


    

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
        elif file_type == '6':
            save_as_SLADE(injected_dataset, filename)
            awaiting_response = False
        elif file_type == '7':
            save_as_SLADE_tail_time_only(injected_dataset, filename)
            awaiting_response = False
        elif file_type == '8':
            save_as_anomrank_with_idle_padding(injected_dataset, filepath)
            awaiting_response = False
            
        else:
            print(f"{YELLOW}Error: Invalid choice! Please try again.{RESET}")

    print(f"{GREEN}Successfully saved data in custom_data folder: exiting...{RESET}")
if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print(f"{YELLOW}\nKeyboard interrupt: exiting...{RESET}")
