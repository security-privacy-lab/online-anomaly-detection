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


def temporal_node_adj(triplet_list):
    tna = dict()
    for u,v,t in triplet_list:
        if u not in tna: tna[u] = dict()
        if t not in tna[u] : tna[u][t] = []
        tna[u][t].append(v)
    return tna
