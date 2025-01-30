import pandas as pd
import random

columns = ['ID', 'Date', 'Time', 'Session_ID', 'Depth', 'Path', 'Label']

def customize_saving_method(edges_df):
    """Allows the user to customize saving columns."""
    new_columns = []
    title = input("What is your desired title for the file? ")
    
    while True:
        question = input(f"What do you want to input? Available columns: {list(edges_df.columns)}\nType 'q' to quit: ")
        if question == 'q':
            break
        elif question in edges_df.columns:
            new_columns.append(question)
            print(f'Current selected columns: {new_columns}')
        else:
            print("Invalid column. Please try again.")
    
    if new_columns:
        edges_df[new_columns].to_csv(f"{title}.csv", sep=',', index=False)
    else:
        print("No columns selected. Exiting.")
    

def load_dataset():
    
    while True: 
        try:
            title = input("Enter the dataset file name (without extension): ")
            dataset_benign = pd.read_csv(f"{title}.txt", sep='|', header=None, names = columns)
            break
        except FileNotFoundError:
            print("The file has not been found. Please try again")
    return dataset_benign
    

def save_as_anomrank_or_f_fade(edges_df):
    """Saves the dataset in AnomRank or F-Fade format."""
    title = input("Enter the desired file name: ")
    edges_df[['timestamp', 'src_node', 'dst_node', 'label']].to_csv(
        f'{title}.txt', sep=' ', header=False, index=False
    )

def save_as_sedanspot(edges_df):
    """Saves the dataset in SedanSpot format."""
    title = input("Enter the desired file name: ")
    edges_df[['timestamp', 'src_node', 'dst_node', 'weight', 'label']].to_csv(
        f'{title}.csv', sep=',', header=False, index=False
    )

def preprocess_malicious_dataset(filename):
    """Preprocesses a malicious dataset from a text file."""
    dataset = load_dataset()
    dataset.drop(["Date", "Time"], axis=1, inplace=True)
    dataset['Label'] = 1
    return dataset

def organized_version(dataset_benign, num_logs):
    """Organizes malicious logs into the dataset in order."""
    for _ in range(num_logs):
        malicious_file_name = input("Enter the malicious file name (without extension): ")
        
        malicious_file = preprocess_malicious_dataset(malicious_file_name)
        dataset_benign = pd.concat((dataset_benign, malicious_file), ignore_index=True)
        dataset_benign = dataset_benign.sort_values(by="Session_ID", ascending=True)
    return dataset_benign

import pandas as pd

def minute_5_gap(injected_dataset, num_logs):
    """Injects malicious logs into 5-minute gaps in session IDs and prints the starting indexes neatly."""
    
    injected_dataset['time_diff'] = injected_dataset['Session_ID'].diff()
    
    gap_indexes = injected_dataset[injected_dataset['time_diff'] >= 300].index.tolist()
    
    if not gap_indexes:
        print("No suitable 5-minute gaps found. Exiting.")
        return injected_dataset
    
    print("\nAvailable gap indexes (start of 5-minute gaps):")
    for i in range(0, len(gap_indexes), 5):
        print("   ".join(f"{idx:>5}" for idx in gap_indexes[i:i+5])) 

    for _ in range(num_logs):
        try: 
            gap_index = int(input("Enter the index of the gap to inject malicious logs: "))
            if gap_index not in gap_indexes:
                print("Invalid index. Please select from the printed list.")
                continue

            malicious_file_name = load_dataset()
            malicious_file = preprocess_malicious_dataset(malicious_file_name)

            injected_dataset = pd.concat([
                injected_dataset.iloc[:gap_index+1],
                malicious_file,
                injected_dataset.iloc[gap_index+1:]
            ]).reset_index(drop=True)

        except ValueError:
            print("Invalid input. Please enter a valid index.")

    print("Malicious logs successfully injected.")
    return injected_dataset


def random_injection(injected_dataset, num_logs):
    """Injects malicious logs randomly into the dataset."""
    malicious_file_name = input("Enter the malicious file name (without extension): ")
    malicious_file = preprocess_malicious_dataset(malicious_file_name)
    injected_indices = []
    
    for _, malicious_row in malicious_file.iterrows():
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
    
    print("Malicious logs successfully injected at random indices.")
    return injected_dataset


dataset_benign = load_dataset()
# Sort by Session_ID
dataset_benign = dataset_benign.sort_values(by='Session_ID').reset_index(drop=True)
dataset_benign['Label'] = 0
injected_dataset = dataset_benign.copy()
choice = input("Choose injection method: rj (random), 5m (5-minute block), or or (organized): ")

if choice in ['rj', '5m', 'or']:
    num_logs = int(input("How many malicious files do you want to inject? "))
    
    if choice == 'rj':
        injected_dataset = random_injection(injected_dataset, num_logs)
    elif choice == '5m':
        injected_dataset = minute_5_gap(injected_dataset, num_logs)
    elif choice == 'or':
        injected_dataset = organized_version(injected_dataset, num_logs)
    
    print(f"Original Dataset Length: {len(dataset_benign)}")
    print(f'Injected Dataset Length: {len(injected_dataset)}')

else: 
    print("Invalid choice. Exiting.")
    exit()


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
    
print("\n************ Generated injected dataset! ************")
print("Choose file save format:")
print("1. Anomrank/F-Fade\n2. Sedanspot\n3. Customize")


edges_df = pd.DataFrame(edges)

file_type = input("Select format (1, 2, or 3):")

if file_type == '1':
    save_as_anomrank_or_f_fade(edges_df)
elif file_type == '2':
    save_as_sedanspot(edges_df)
elif file_type == '3':
    customize_saving_method(edges_df)
else:
    print("Invalid selection. Exiting.")
