import csv
def read_data(data_path):
    data = []
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader: 
            data.append( (row[0], row[1], int(row[2])) )
    return data

def get_stats(dataset):
    ts_graphs = dict()
    V = set()
    for u,v,t in dataset: 
        V.add(u); V.add(v);
        if t not in ts_graphs: ts_graphs[t] = []
        ts_graphs[t] += [(u,v)]
       
    triplets = len(dataset)
    nodes = len(V)
    max_time = max(ts_graphs.keys())
    empty_slices = len([t for t in range(max_time) if t not in ts_graphs])
    activity_peak = max([len(ts_graphs[t]) for t in ts_graphs])

    return triplets, nodes, max_time, empty_slices, activity_peak



synthetic_data = read_data('./synthetic.txt')
hospital_data = read_data('./hospital.txt')
emails_data = read_data('./emails.txt')
traffic_data = read_data('./traffic.txt')

triplets, nodes, max_time, empty_slices, activity_peak = get_stats(synthetic_data)
print('------ Synthetic dataset stats --------')
print('Triplets: ', triplets)
print('Nodes: ', nodes)
print('Max time: ', max_time)
print('Empty slices: ', empty_slices)
print('Activity peak: ', activity_peak)

triplets, nodes, max_time, empty_slices, activity_peak = get_stats(hospital_data)
print('------ Hospital dataset stats --------')
print('Triplets: ', triplets)
print('Nodes: ', nodes)
print('Max time: ', max_time)
print('Empty slices: ', empty_slices)
print('Activity peak: ', activity_peak)

triplets, nodes, max_time, empty_slices, activity_peak = get_stats(emails_data)
print('------ Emails dataset stats --------')
print('Triplets: ', triplets)
print('Nodes: ', nodes)
print('Max time: ', max_time)
print('Empty slices: ', empty_slices)
print('Activity peak: ', activity_peak)

triplets, nodes, max_time, empty_slices, activity_peak = get_stats(traffic_data)
print('------ Traffic dataset stats --------')
print('Triplets: ', triplets)
print('Nodes: ', nodes)
print('Max time: ', max_time)
print('Empty slices: ', empty_slices)
print('Activity peak: ', activity_peak)

