import pandas as pd 
import random 
import hashlib 

columns = ['ID', 'Date', 'Time', 'Session_ID', 'Depth', 'Path', 'Label']
# Different saving methods
def save_as_anomrank_or_f_fade(edges_df, title):
    edges_df[['timestamp','src_node', 'dst_node', 'label']].to_csv(
    f'{title}.txt', sep = ' ', header = False, index = False
    )
def save_as_sedanspot(edges_df, title):
    edges_df[['timestamp','src_node', 'dst_node', 'weight', 'label']].to_csv(
    f'{title}.csv', sep = ',', header = False, index = False
    )
def preprocess_malicious_dataset(filename):
    dataset = pd.read_csv(f"{filename}.txt", sep = '|', header = None, names = columns)
    dataset.drop(["Date", "Time"], axis = 1, inplace = True)
    dataset['Label'] = 1
    return dataset

def organized_version(dataset_benign):
    num_logs = int(input("How many attack logs do you want to insert? :"))
    for _ in range(num_logs):
        malicious_file_name = input("Enter the malicious file name(Expecting the extension as '.txt' )")
        malicious_file = preprocess_malicious_dataset(malicious_file_name)
        dataset_benign = pd.concat((injected_dataset, malicious_file),ignore_index= True)
        dataset_benign = dataset_benign.sort_values(by="Session_ID", ascending=True)
    return dataset_benign

def minute_5_gap(injected_dataset):
    
    injected_dataset['time_diff'] = injected_dataset['Session_ID'].diff()
    gaps = injected_dataset[injected_dataset['time_diff'] >= 300].reset_index()
    if gaps.empty:
        print("No suitable 5-minute gaps found. Exiting.")
        return None
    num_logs = int(input("How many attack logs do you want to insert? :"))

    for _ in range(num_logs):
        print(f"\nAvailable gaps:\n{gaps[['Session_ID']].to_string(index=True)}")
        gap_index = int(input("Enter the index of the gap to inject malicious logs: "))
        malicious_file_name = input("Enter the malicious file name(Expecting the extension as '.txt' )")
        malicious_file = preprocess_malicious_dataset(malicious_file_name)
        injected_dataset = pd.concat([
            injected_dataset.iloc[:gap_index+1],
            malicious_file,
            injected_dataset.iloc[gap_index+1:]
        ]).reset_index(drop=True)
    else:
        print("Invalid malicious log choice!")
        exit()    
    print("Malicious logs successfully injected")
    return injected_dataset

def random_injection(injected_dataset):
    try: 
        malicious_data = input("Please enter the name of the malicious file you are trying to insert. [[DO NOT WRITE .TXT on the end, just the name of the file!!!!]] ")
    except FileNotFoundError:
        print(f"file {malicious_data} was not found. Please try again")
        exit()
    malicious_file = preprocess_malicious_dataset(malicious_data)   
    
    injected_indices = [] # this is for checking if the dataset really have been randomly inserted

    for _, malicious_row in malicious_file.iterrows(): # We read the malicious dataset here
            malicious_data = {
                    'Session_ID': malicious_row['Session_ID'],
                    'Depth': malicious_row['Depth'],
                    'Path': malicious_row['Path'],
                    'Label': malicious_row['Label']
        }
            malicious_row_df = pd.DataFrame([malicious_data])
            
            random_index = random.randint(0, len(injected_dataset)) # To make sure that the malicious are being injected to random index 
            injected_indices.append(random_index)
            injected_dataset = pd.concat([
                    injected_dataset.iloc[:random_index],  
                    malicious_row_df,                      
                    injected_dataset.iloc[random_index:]  
            ]).reset_index(drop=True)
    
    return injected_dataset


file_name = input("input the file name without the extension (Expecting the input file to be txt.)")
dataset_benign = pd.read_csv(f"{file_name}.txt", sep='|', header=None, names=columns)

# Sort by Session_ID and reset index
dataset_benign = dataset_benign.sort_values(by='Session_ID').reset_index(drop=True)
dataset_benign['Label'] = 0

choice = input("Are you trying to make Random injection or 5-minute block injection? \nAnswer with: rj (random injection) or 5m (5-minute injection): ")
injected_dataset = dataset_benign.copy()

if choice in ['rj', '5m', 'or']:
    if choice == 'rj':
        numbers_of_malicious_files_injected = input("How many malicious files do you want to inject?")
        for _ in range(int(numbers_of_malicious_files_injected)):
            injected_dataset = random_injection(injected_dataset)
        print(f"Original Dataset Length : {len(dataset_benign)}")
        print(f'Currnet dataset length : {len(injected_dataset)}')
    elif choice == '5m':
        numbers_of_malicious_files_injected = input("How many malicious files do you want to inject?")
        for _ in range(int(numbers_of_malicious_files_injected)):
            injected_dataset = minute_5_gap(injected_dataset)
        print(f"Original Dataset Length : {len(dataset_benign)}")
        print(f'Current dataset length : {len(injected_dataset)}')
    elif choice == 'or':
        exit()
    else: 
        print("Invalid choice, exiting...")
        exit()
    
print("************ Generated injected dataset! ************")
print("How do you want your file to be saved?")
print("Please type the numbers")

edges = []
prev_path = None

for _, row in injected_dataset.iterrows():
    current_path = row['Path']
    timestamp = row['Session_ID']
    label = row['Label']

    # Self-edge
    if prev_path == current_path:
        edges.append({
            'src_node' : current_path,
            'dst_node': current_path,
            'timestamp' : timestamp,
            'weight' : 1,
            'label' : label
            
        })
    elif prev_path is not None:
        edges.append({      
            'src_node' : prev_path,
            'dst_node': current_path,
            'timestamp' : row['Session_ID'],
            'weight' : 1, # Keeps the weight as 1 as default 
            'label' : label
        })
    prev_path = current_path

edges_df = pd.DataFrame(edges)

file_type = input("1. Anomrank/F-Fade\n2. Sedanspot.")
if file_type == '1':
    save_as_anomrank_or_f_fade(edges_df)
elif file_type == '2':
    save_as_sedanspot(edges_df)
