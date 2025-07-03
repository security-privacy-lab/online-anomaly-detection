import pandas as pd
import os

COLUMNS = ['ID', 'Date', 'Time', 'Session_ID', 'Depth', 'Path']
SEPARATOR = '|'

def load_log_interactively(label_desc):
    while True:
        filename = input(f"Enter the {label_desc} dataset filename: ")
        if not filename.endswith('.txt'):
            filename += '.txt'
        if not os.path.isfile(filename):
            print(f"File '{filename}' not found. Please try again.")
        else:
            df = pd.read_csv(filename, sep=SEPARATOR, header=None, names=COLUMNS)
            df['Label'] = 0 if label_desc == 'benign' else 1
            return df

def find_5min_gaps(df):
    df['Session_ID'] = df['Session_ID'].astype(int)
    diffs = df['Session_ID'].diff().abs().fillna(0)
    return diffs[diffs >= 301].index.tolist()

def get_quartile_groups(indices, total_len):
    quartiles = {
        '0-25%': [],
        '25-50%': [],
        '50-75%': [],
        '75-100%': []
    }
    for idx in indices:
        pct = (idx / total_len) * 100
        if pct <= 25:
            quartiles['0-25%'].append(idx)
        elif pct <= 50:
            quartiles['25-50%'].append(idx)
        elif pct <= 75:
            quartiles['50-75%'].append(idx)
        else:
            quartiles['75-100%'].append(idx)
    return quartiles

def print_quartile_gap_indices(quartiles):
    print("\nAvailable 5-minute gap indexes divided into quartile groups:")
    for q_range, indices in quartiles.items():
        index_preview = ', '.join(map(str, indices))
        print(f"  {q_range} block: {index_preview} ")

def inject_at_index(benign_df, malicious_df_list, gap_indices):
    for mal_df in malicious_df_list:
        quartiles = get_quartile_groups(gap_indices, len(benign_df))
        print_quartile_gap_indices(quartiles)

        while True:
            try:
                gap_idx = int(input("Enter the index to inject malicious logs: "))
                if gap_idx not in gap_indices:
                    print("Invalid index. Please choose from the listed gap indices.")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer.")

        approx_pct = (gap_idx / len(benign_df)) * 100
        print(f"Selected index {gap_idx} is approximately at {approx_pct:.2f}% of the file.")

        base_id = int(mal_df.iloc[0]['Session_ID'])
        mal_df['Session_ID'] = mal_df['Session_ID'].astype(int) - base_id + int(benign_df.loc[gap_idx, 'Session_ID']) + 1

        benign_df = pd.concat([
            benign_df.iloc[:gap_idx + 1],
            mal_df,
            benign_df.iloc[gap_idx + 1:]
        ]).reset_index(drop=True)

        gap_indices = [idx + len(mal_df) if idx > gap_idx else idx for idx in gap_indices if idx != gap_idx]

    return benign_df

def inject_organized(benign_df, malicious_df_list):
    for mal_df in malicious_df_list:
        benign_df = pd.concat([benign_df, mal_df])
    return benign_df.sort_values(by='Session_ID').reset_index(drop=True)

def save_log(df, filename):
    if not filename.endswith('.txt'):
        filename += '.txt'
    df[COLUMNS].to_csv(filename, sep=SEPARATOR, header=False, index=False)
    print(f"Injected dataset saved to {filename}")

def main():
    benign_df = load_log_interactively("benign")

    while True:
        try:
            n = int(input("How many malicious files would you like to input: "))
            break
        except ValueError:
            print("Please enter a valid integer.")

    malicious_list = []
    for i in range(n):
        mal_df = load_log_interactively(f"malicious file #{i+1}")
        malicious_list.append(mal_df)

    method = input("Which method to inject (5m or org): ").lower().strip()
    if method == "5m":
        gaps = find_5min_gaps(benign_df)
        if not gaps:
            print("No suitable 5-minute gaps found.")
            return
        result_df = inject_at_index(benign_df, malicious_list, gaps)
    elif method == "org":
        result_df = inject_organized(benign_df, malicious_list)
    else:
        print("Invalid method. Choose either '5m' or 'org'.")
        return

    out_file = input("Enter output file name: ")
    save_log(result_df, out_file)

if __name__ == "__main__":
    main()
