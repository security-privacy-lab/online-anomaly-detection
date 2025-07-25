import csv

def read_data(data_path):
    data = []
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader: 
            data.append( (row[0], row[1], int(row[2])) )
    return data

def save_data(data_path, data_list):
    with open(data_path, 'w', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(data_list)


def time_series_graphs(triplet_list):
    tsg = dict()
    for u,v,t in triplet_list:
        if t not in tsg : tsg[t] = []
        tsg[t].append( (u,v) )
    return tsg
