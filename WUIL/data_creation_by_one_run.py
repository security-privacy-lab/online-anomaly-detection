import pandas as pd
import random
import hashlib
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
    """Injects malicious logs into 5-minute gaps in session IDs and prints the starting indexes neatly."""

    # Correct input to type list
    if not isinstance(malicious_dataset, list):
        malicious_dataset = [malicious_dataset]

    injected_dataset = benign_dataset.copy()

    difference = pd.DataFrame([], columns= ["time_diff"])
    difference['time_diff'] = injected_dataset['Session_ID'].astype(int).shift(-1) - injected_dataset['Session_ID'].astype(int)
    gap_indexes = difference[difference['time_diff'] >= 301].index.tolist()

    
    if not gap_indexes:
        raise("No suitable 5-minute gaps found. Exiting.")


    while malicious_dataset:
            try: 
                print("\nAvailable gap indexes (start of 5-minute gaps):")
                for i in range(0, len(gap_indexes), 10):
                    print("   ".join(f"{idx:>5}" for idx in gap_indexes[i:i+10])) 
                gap_index = int(input("Enter the index of the gap to inject malicious logs: "))
                
                if gap_index not in gap_indexes:
                    print(f"{YELLOW}Error: Invalid choice! Please enter a valid index.{RESET}")
                    continue

                malicious_file = malicious_dataset.pop(0)

                malicious_file["Session_ID"] = malicious_file["Session_ID"].astype(int) - int(malicious_file.loc[0, "Session_ID"]) + 1 + int(injected_dataset.loc[gap_index, "Session_ID"])

                injected_dataset = pd.concat([
                    injected_dataset.iloc[:gap_index+1],
                    malicious_file,
                    injected_dataset.iloc[gap_index+1:]
                ]).reset_index(drop=True)

                gap_indexes = [i+len(malicious_file) for i in gap_indexes if i != gap_index]
            except ValueError:
                print(f"{YELLOW}Error: Invalid choice! Please enter a valid index.{RESET}")

    print(f"{GREEN}Malicious logs successfully injected.{RESET}")
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


def save_as_MAD(injected_data, filepath, malicious_dataset):
    edges = []
    prev_path = None

    for _, row in injected_data.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']
        current_path_hashed = int(hashlib.md5(current_path.encode()).hexdigest(), 16) % (10**8)

        # Self-edge
        if prev_path == current_path:
            edges.append({
                'src_node' : current_path_hashed,
                'dst_node': current_path_hashed,
                'timestamp' : timestamp,
                'label' : label
                
            })
        elif prev_path is not None:
            prev_path_hashed = int(hashlib.md5(prev_path.encode()).hexdigest(), 16) % (10**8)
            edges.append({      
                'src_node' : prev_path_hashed,
                'dst_node': current_path_hashed,
                'timestamp' : row['Session_ID'],
                'label' : label
            })
        prev_path = current_path
    edges_df = pd.DataFrame(edges)
    edges_df[['timestamp', 'src_node', 'dst_node', 'label']].to_csv(
        f'{filepath}_dataset.txt', sep=',', header=False, index=False
    )
    dataset_filtered = edges_df[edges_df['label'] == 1]

    query = dataset_filtered[['src_node', 'dst_node']].drop_duplicates()
    query.to_csv(f'{filepath}_queries.txt',sep=',', header=False,index=False)

    dataset = edges_df.copy()
    query_nodes = set(query['src_node']).union(set(query['dst_node']))
    related_edges = dataset[(dataset['src_node'].isin(query_nodes)) | (dataset['dst_node'].isin(query_nodes))]
    related_edges.to_csv(f"{filepath}_gt.txt",sep=',', index=False,header=None)






def save_as_anomrank_or_f_fade(injected_dataset, filepath):
    edges = []
    prev_path = None

    for _, row in injected_dataset.iterrows():
        current_path = row['Path']
        timestamp = row['Session_ID']
        label = row['Label']
        current_path_hashed = int(hashlib.md5(current_path.encode()).hexdigest(), 16) % (10**8)

        # Self-edge
        if prev_path == current_path:
            edges.append({
                'src_node' : current_path_hashed,
                'dst_node': current_path_hashed,
                'timestamp' : timestamp,
                'label' : label
                
            })
        elif prev_path is not None:
            prev_path_hashed = int(hashlib.md5(prev_path.encode()).hexdigest(), 16) % (10**8)
            edges.append({      
                'src_node' : prev_path_hashed,
                'dst_node': current_path_hashed,
                'timestamp' : row['Session_ID'],
                'label' : label
            })
        prev_path = current_path

    edges_df = pd.DataFrame(edges)
    """Saves the dataset in AnomRank or F-Fade format."""
    edges_df[['timestamp', 'src_node', 'dst_node', 'label']].to_csv(
        f'{filepath}', sep=' ', header=False, index=False
    )

def save_as_sedanspot(edges_df, filepath):
    """Saves the dataset in SedanSpot format."""
    edges_df[['timestamp', 'src_node', 'dst_node', 'weight', 'label']].to_csv(
        f'{filepath}.csv', sep=',', header=False, index=False
    )

def save_as_midas(edges_df, filepath):
    """Saves the dataset in Midas format."""
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
        choice = input(f"Choose injection method: {CYAN}rj{RESET} (random), {CYAN}5m{RESET} (5-minute block), or {CYAN}or{RESET} (organized): ")
        
        if choice in ['rj', '5m', 'or']:

            num_logs = int(input("How many malicious files do you want to inject? : "))
            
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
            save_as_MAD(injected_dataset, filepath, malicious_dataset)
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
