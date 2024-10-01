import pandas as pd
import numpy as np
import subprocess
import os
import argparse
from sklearn.metrics import roc_auc_score


def generate_minute_data(folder):

    if os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    data_folder = os.path.join(folder, "CIC_minute_data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    label_folder = os.path.join(folder, "CIC_minute_label")
    if not os.path.exists(label_folder):
        os.makedirs(label_folder, exist_ok=True)

    df = pd.read_csv("Darknet.CSV")

    print(df[""])

    # format the time column
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p")

    # create new minute timestamp column
    df["timestamp_minute"] = df["Timestamp"].dt.floor("min")

    grouped = df.groupby("timestamp_minute")

    for timestamp_minute, group in grouped:
        # create a grp_df with the "Src IP" and "Dst IP" columns for SedanSpot

        group = group.sort_values(by=["Timestamp"])

        grp_df = group[["Src IP", "Dst IP"]].copy()

        # add "weight" feature to grp_df
        grp_df.loc[:, "weight"] = np.ones(grp_df.shape[0]).astype(int)

        # convert categorical data "Label" to 0 or 1
        grp_df.loc[:, "Label Condition"] = np.where(
            (group["Label"].str.lower() == "nonvpn") | (group["Label"].str.lower() == "non-tor"), 0, 1)

        label_df = grp_df["Label Condition"].copy()

        # export the new df into a csv file
        grp_df.to_csv(os.path.join(
            data_folder, f"{timestamp_minute}_data.csv"), index=False, header=False)

        label_df.to_csv(os.path.join(
            label_folder, f"{timestamp_minute}_label.csv"), index=False, header=False)


def generate_cont_data(folder):

    if os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    df = pd.read_csv("Darknet.CSV")

    # format the time column
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], format="%d/%m/%Y %I:%M:%S %p")

    # create new minute timestamp column
    df["timestamp_minute"] = df["Timestamp"].dt.floor("min")

    df = df.sort_values(by=["Timestamp"])

    new_df = df[["Src IP", "Dst IP"]].copy()

    # add "weight" feature to grp_df
    new_df.loc[:, "weight"] = np.ones(new_df.shape[0]).astype(int)

    # convert categorical data "Label" to 0 or 1
    new_df.loc[:, "Label Condition"] = np.where(
        (df["Label"].str.lower() == "nonvpn") | (df["Label"].str.lower() == "non-tor"), 0, 1)
  
    grouped = df.groupby(
        "timestamp_minute").size().reset_index(name='count')

    grouped.sort_values(by="timestamp_minute")

    new_df.to_csv(os.path.join(folder, f"CIC_cont_data.csv"),
                  index=False, header=False)

    new_df["Label Condition"].to_csv(os.path.join(folder, f"CIC_cont_label.csv"),
                                     index=False, header=False)

    grouped.to_csv(os.path.join(folder, f"groups.csv"),
                   index=False, header=False)

def run_cont_test():

    folder = "CIC_cont"

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    generate_cont_data(folder)

    file_path = os.path.join(folder, "CIC_cont_data.csv")

    print(pd.read_csv(file_path).shape[0])
    
    if os.path.isfile(file_path):
        output_file = os.path.join(
            folder, "CIC_cont_output.csv")

        with open(output_file, 'w') as f:
            subprocess.run(
                ["./bin/SedanSpot", "--input", file_path], stdout=f)
    
    probs = pd.read_csv(output_file).iloc[:, 0].to_numpy()
    labels = pd.read_csv(os.path.join(folder, "CIC_cont_label.csv")).iloc[:, 0].to_numpy()

    print(labels.shape, probs.shape)
    # print(roc_auc_score(labels.tolist(), probs.tolist()))
    

def run_minute_test():
    folder = "CIC_minute"

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    generate_minute_data(folder)

    dir_path = os.path.join(folder, "CIC_minute_data")

    output_path = os.path.join(folder, "CIC_minute_output")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            output_file = os.path.join(
                output_path, f"{filename}_output.csv")

            with open(output_file, 'w') as f:
                subprocess.run(
                    ["./bin/SedanSpot", "--input", file_path], stdout=f)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="process arguments to run either minute-based test or continuous test")

    parser.add_argument(
        '--type', type=str, help='the type of test to run, continuous or minute', choices=["continuous", "minute"])

    args = parser.parse_args()

    if args.type == "continuous":
        run_cont_test()
    else:
        run_minute_test()
