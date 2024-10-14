#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include "time.h"

// Include the required files (adjust if necessary)
#include "accuracy.cpp"
#include "anomaly_detect.cpp"
#include "anomaly_inject.cpp"
#include "pagerank.cpp"
#include "read_data.cpp"

#define attackLimit 110

using namespace std;

int main(int argc, char *argv[]) {
    clock_t start = clock();

    string path = argv[1];
    string delimiter = argv[2];
    int timeStep = atoi(argv[3]);
    int initSS = atoi(argv[4]);
    int injectScene = atoi(argv[2]);
    int injectNum = atoi(argv[6]);
    int injectSize = atoi(argv[7]);
    bool INJECT = (injectScene != 0);

    // Read data
    vector<timeEdge> edges;
    vector<int> snapshots;
    int n, m, timeNum;
    read_data(path, delimiter, timeStep, edges, snapshots, n, m, timeNum);
    int numSS = timeNum / timeStep + 1;
    outEdge* A = new outEdge[n];
    cout << "#node: " << n << ", #edges: " << edges.size() << ", #timeStamp: " << timeNum << endl;

    // Anomaly score setup
    int testNum = numSS - initSS;
    bool* attack = new bool[testNum];
    double* anomScore = new double[testNum];
    for (int i = 0; i < testNum; i++) {
        anomScore[i] = 0;
        attack[i] = false;
    }

    // PageRank setup
    double** pagerank1 = new double*[3];
    double** pagerank2 = new double*[3];
    for (int i = 0; i < 3; i++) {
        pagerank1[i] = new double[n];
        pagerank2[i] = new double[n];
        for (int j = 0; j < n; j++) {
            pagerank1[i][j] = pagerank2[i][j] = 0;
        }
    }

    // Mean and Variance setup
    double** mean = new double*[4];
    double** var = new double*[4];
    for (int i = 0; i < 4; i++) {
        mean[i] = new double[n];
        var[i] = new double[n];
        for (int j = 0; j < n; j++) {
            mean[i][j] = var[i][j] = 0;
        }
    }

    // Injected snapshot setup
    vector<int> injectSS;
    if (INJECT)
        inject_snapshot(injectNum, initSS, testNum, snapshots, injectSS);

    cout << "Preprocess done: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl;

    // Processing each snapshot and computing scores
    int eg = 0;
    int snapshot = 0;
    int attackNum = 0;
    int injected = 0;
    int current_m = 0;
    double previous_score = 100.0;

    start = clock();
    int print_e = 10;
    for (int ss = 0; ss < snapshots.size(); ss++) {
        while (edges[eg].t < snapshots[ss] * timeStep) {
            inject(A, edges[eg].src, edges[eg].trg, 1);
            current_m++;
            if (edges[eg].atk)
                attackNum++;
            eg++;
            if (eg == print_e) {
                cout << eg << "," << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
                print_e *= 10;
            }
            if (eg == edges.size())
                break;
        }

        if (INJECT && injectSS[injected] == snapshots[ss]) {
            current_m += inject_anomaly(injectScene, A, n, injectSize);
            attackNum += attackLimit;
            injected++;
            if (injected == injectSS.size())
                INJECT = false;
        }

        snapshot = snapshots[ss];
        pagerank(A, pagerank1[snapshot % 3], n, current_m, 1);
        pagerank(A, pagerank2[snapshot % 3], n, current_m, 2);

        double score = compute_anomaly_score(snapshot, pagerank1, pagerank2, mean, var, n);
        if (snapshot >= initSS) {
            anomScore[snapshot - initSS] = score;
            attack[snapshot - initSS] = attackNum >= attackLimit;
            previous_score = score;
        }
        attackNum = 0;
    }

    // Write anomaly scores to a file
    string filePath = "darpa_anomrank.txt";
    ofstream writeFile;
    writeFile.open(filePath.c_str(), ofstream::out);
    for (int i = 0; i < testNum; i++)
        writeFile << anomScore[i] << " " << int(attack[i]) << endl;
    writeFile.close();

    // Compute accuracy and store precision-recall data
    vector<double> precisions;
    vector<double> recalls;
    for (int i = 1; i <= 16; i++) {
    double precision = 0.0;
    double recall = 0.0;

    // Ensure only one call to compute_accuracy
    compute_accuracy(anomScore, attack, testNum, 50 * i, precision, recall);

    // Ensure only one print statement for each calculation
    cout << "Top " << 50 * i << " - Precision: " << precision << ", Recall: " << recall << endl;

    // Store precision and recall for plotting if needed
    if (i <= 5) {
        precisions.push_back(precision);
        recalls.push_back(recall);
        }
    }

    

    // Output precision and recall data for plotting
    ofstream plotFile("precision_recall_data.txt");
    for (size_t i = 0; i < precisions.size(); ++i) {
        plotFile << (50 * (i + 1)) << " " << precisions[i] << " " << recalls[i] << endl;
    }
    plotFile.close();

    // Free allocated memory
    delete[] A;
    delete[] anomScore;
    delete[] attack;

    for (int i = 0; i < 3; i++) {
        delete[] pagerank1[i];
        delete[] pagerank2[i];
    }
    delete[] pagerank1;
    delete[] pagerank2;

    for (int i = 0; i < 4; i++) {
        delete[] mean[i];
        delete[] var[i];
    }
    delete[] mean;
    delete[] var;

    return 0;
}
