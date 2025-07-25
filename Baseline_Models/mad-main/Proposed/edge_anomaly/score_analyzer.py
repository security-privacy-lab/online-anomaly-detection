import csv
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def read_score_file(data_path):
    data_dict = dict()
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader: 
            u = row[0]
            v = row[1]
            t = row[2]
            score = float(row[3])
            data_dict[(u,v,t)] = score
    return data_dict


if __name__ == '__main__':

    # Parse commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    score_file = './result/' + args.dataset + '_scores.txt'
    label_file = './labels/' + args.dataset + '_gt.txt'
    roc_file = './ROC_AUC/' + args.dataset + '_roc.txt'

    # Read algo scores
    scores_dict = read_score_file(score_file)

    # Read gt
    labels_dict = read_score_file(label_file)

    # Compare scores output by algo with labels
    algo = []
    label = []
    label_norm = []
    fake = []
    for q in scores_dict:
        algo.append( scores_dict[q] )
        label.append( int(labels_dict[q]) )
        label_norm.append( labels_dict[q]*scores_dict[q] )

    ROC_AUC = roc_auc_score(label, algo)
    print('Dataset : ', args.dataset ,' ; ROC AUC : ', ROC_AUC)

    # Save result
    np.savetxt(roc_file, [ROC_AUC], fmt='%f')
