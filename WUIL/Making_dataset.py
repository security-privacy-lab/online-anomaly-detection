import pandas as pd
import random

# Define column headers
columns = ['ID', 'Date', 'Time', 'Session_ID', 'Depth', 'Path', 'Label']

# Step 1: Load benign dataset
print('Enter which user file you would like to use: \nChoices: \n1 \n2 \n3')
choice = input("Choose the user file from 1/2/3: ")
if choice in ['1', '2', '3']:
    dataset_benign = pd.read_csv(f"user{choice}_log.txt", sep='|', header=None, names=columns)
    print("Benign dataset loaded successfully!")
else:
    print("Invalid choice! Exiting.")
    exit()

# Sort by Session_ID and reset index
dataset_benign = dataset_benign.sort_values(by='Session_ID').reset_index(drop=True)
dataset_benign['Label'] = 0

# Step 2: Load malicious datasets
malicious_logs1 = pd.read_csv('Attack1_log.txt', sep='|', header=None, names=columns)
malicious_logs2 = pd.read_csv('Attack2_log.txt', sep='|', header=None, names=columns)
malicious_logs3 = pd.read_csv('Attack3_log.txt', sep='|', header=None, names=columns)
combined_malicious_logs = pd.concat([malicious_logs1, malicious_logs2, malicious_logs3], ignore_index=True)

# Step 3: Inject malicious logs
choice = input("Are you trying to make Random injection or 5-minute block injection? \nAnswer with: rj (random injection) or 5m (5-minute injection): ")

injected_dataset = dataset_benign.copy()

if choice in ['rj', '5m']:
    if choice == 'rj':
        max_size = len(combined_malicious_logs)
        parameter = int(input(f"Enter the number of malicious rows to inject (max {max_size}): "))
        if parameter > max_size:
            print("Invalid size! Exiting.")
            exit()
        else:
            malicious_subset = combined_malicious_logs.sample(parameter, random_state=123)

    elif choice == '5m':
        dataset_benign['time_diff'] = dataset_benign['Session_ID'].diff()
        gaps = dataset_benign[dataset_benign['time_diff'] >= 3000].reset_index()
        if gaps.empty:
            print("No suitable 5-minute gaps found. Exiting.")
            exit()

        num_logs = int(input("How many attack logs do you want to insert? (1, 2, or 3): "))
        for _ in range(num_logs):
            print(f"\nAvailable gaps:\n{gaps[['Session_ID']].to_string(index=False)}")
            gap_index = int(input("Enter the index of the gap to inject malicious logs: "))
            malicious_input = input("Choose a malicious log to inject (1, 2, or 3): ")
            if malicious_input in ['1', '2', '3']:
                malicious_log = pd.read_csv(f'Attack{malicious_input}_log.txt', sep='|', header=None, names=columns)
                injected_dataset = pd.concat([
                    injected_dataset.iloc[:gap_index],
                    malicious_log,
                    injected_dataset.iloc[gap_index:]
                ]).reset_index(drop=True)
            else:
                print("Invalid malicious log choice!")
                exit()

    # Inject malicious rows randomly for `rj` or at gaps for `5m`
    for _, malicious_row in malicious_subset.iterrows():
        random_index = random.randint(0, len(injected_dataset))
        malicious_data = {
            'ID': malicious_row['ID'],
            'Session_ID': malicious_row['Session_ID'],
            'Depth': malicious_row['Depth'],
            'Path': malicious_row['Path'],
            'Label': 1
        }
        malicious_row_df = pd.DataFrame([malicious_data])
        injected_dataset = pd.concat([
            injected_dataset.iloc[:random_index],
            malicious_row_df,
            injected_dataset.iloc[random_index:]
        ]).reset_index(drop=True)

    print(f"Injected {len(malicious_subset)} malicious rows into the benign dataset.")
else:
    print("Invalid choice! Exiting.")
    exit()

# Step 4: Create edges with optional weights
answer = input("Do you want to consider the weights too? (y/n): ")
edges = []
edge_weights = {}

if answer == 'y':
    prev_path = None
    for _, row in injected_dataset.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']

        if prev_path == current_path:
            edge_key = (current_path, current_path)
        elif prev_path is not None:
            edge_key = (prev_path, current_path)
        else:
            prev_path = current_path
            continue

        edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
        edges.append({
            'src_node': edge_key[0],
            'dst_node': edge_key[1],
            'timestamp': timestamp,
            'weight': edge_weights[edge_key],
            'label': label
        })
        prev_path = current_path
else:
    prev_path = None
    for _, row in injected_dataset.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']

        if prev_path == current_path:
            edges.append({'src_node': current_path, 'dst_node': current_path, 'timestamp': timestamp, 'label': label})
        elif prev_path is not None:
            edges.append({'src_node': prev_path, 'dst_node': current_path, 'timestamp': timestamp, 'label': label})
        prev_path = current_path

# Save edges to file
edges_df = pd.DataFrame(edges)
file_name = input("How do you want to name your dataset file? ")
edges_df.to_csv(f"{file_name}.csv", index=False)

print("\nDone! The dataset with injected traffic has been saved.")
