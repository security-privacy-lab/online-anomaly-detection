import csv

def read_data(data_path):
    data = []
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader: 
            data.append( (row[0], row[1], int(row[2])) )
    return data

def save_data(data_path, data_list, newline=''):
    with open(data_path, 'w') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(data_list)


def edge_timets(triplet_list):
    ets = dict()
    for u,v,t in triplet_list:
        if (u,v) not in ets: ets[(u,v)] = []
        ets[(u,v)].append( t ) 
    return ets
