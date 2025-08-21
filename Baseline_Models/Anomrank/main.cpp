#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>
#include "time.h"
#include <string>       // Required for std::to_string
#include <sstream>      // Required for std::stringstream

#include <algorithm>
#include "accuracy.cpp"
#include "anomaly_detect.cpp"
#include "anomaly_inject.cpp"
#include "pagerank.cpp"
#include "read_data.cpp"

#define attackLimit 5

using namespace std;

int main(int argc, char *argv[])
{
    clock_t start = clock();

    string path = argv[1];
    string delimeter = argv[2];
    int timeStep = atoi(argv[3]);
    int initSS = atoi(argv[4]);
    int injectScene = atoi(argv[5]);
    int injectNum = atoi(argv[6]);
    int injectSize = atoi(argv[7]);
    bool INJECT = (injectScene != 0);

    // READ DATA
    vector<timeEdge> edges;
    vector<int> snapshots;
    int n, m, timeNum;
    read_data(path, delimeter, timeStep, edges, snapshots, n, m, timeNum);
    int numSS = timeNum/timeStep + 1;
    outEdge* A = new outEdge[n];
    cout << "#node: " << n << ", #edges: "<< edges.size() << ", #timeStamp: " << timeNum << endl;

    // ANOMALY_SCORE
    int testNum = numSS - initSS;
    bool* attack = new bool[testNum];
    double* anomScore = new double[testNum];
    for(int i = 0; i < testNum; i++)
    {
        anomScore[i] = 0;
        attack[i] = false;
    }

    // PAGERANK
    double** pagerank1 = new double*[3];
    double** pagerank2 = new double*[3];
    for(int i = 0; i < 3; i++)
    {
        pagerank1[i] = new double[n];
        pagerank2[i] = new double[n];
        for(int j = 0; j < n; j++)
            pagerank1[i][j] = pagerank2[i][j] = 0;
    }

    // MEAN AND VARIANCE
    double** mean = new double*[4];
    double** var = new double*[4];
    for(int i = 0; i < 4; i++)
    {
        mean[i] = new double[n];
        var[i] = new double[n];
        for(int j = 0; j < n; j++)
            mean[i][j] = var[i][j] = 0;
    }

    // INJECTED SNAPSHOT
    vector<int> injectSS;
    if(INJECT)
        inject_snapshot(injectNum, initSS, testNum, snapshots, injectSS);

    cout << "Preprocess done: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl;

    int eg = 0;
    int snapshot = 0;
    int attackNum = 0;
    int injected = 0;
    int current_m = 0;
    double previous_score = 100.0;

    start = clock();
    int print_e = 10;
    for(int ss = 0; ss < snapshots.size(); ss++)
    {
        while(eg < edges.size() && edges[eg].t < snapshots[ss]*timeStep)
        {
            inject(A, edges[eg].src, edges[eg].trg, 1);
            current_m++;
            if(edges[eg].atk)
                attackNum++;
            eg++;
            if(eg == print_e)
            {
                cout << eg << "," << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
                print_e *= 10;
            }
        }

        if(INJECT && !injectSS.empty() && injected < injectSS.size() && injectSS[injected] == snapshots[ss])
        {
            current_m += inject_anomaly(injectScene, A, n, injectSize);
            attackNum += attackLimit;
            injected++;
            if(injected == injectSS.size())
                INJECT = false;
        }

        snapshot = snapshots[ss];
        pagerank(A, pagerank1[snapshot%3], n, current_m, 1);
        pagerank(A, pagerank2[snapshot%3], n, current_m, 2);

        double score = compute_anomaly_score(snapshot, pagerank1, pagerank2, mean, var, n);
        if(snapshot >= initSS)
        {
            int anom_idx = snapshot - initSS;
            if (anom_idx >= 0 && anom_idx < testNum)
            {
                anomScore[anom_idx] = score;
                attack[anom_idx] = attackNum >= attackLimit;
                previous_score = score;
            }
        }
        attackNum = 0;
    }

    // --- Build output file path from input file name ---
    string base_name = path;
    size_t last_slash = base_name.find_last_of("/\\");
    if (string::npos != last_slash)
    {
        base_name.erase(0, last_slash + 1);
    }
    
    size_t extension_pos = base_name.find_last_of(".");
    if (string::npos != extension_pos)
    {
        base_name.erase(extension_pos);
    }

    // WRITE ANOMALY SCORE
    string anom_filePath = base_name + "_anomrank_result.txt";
    ofstream anom_writeFile;
    anom_writeFile.open(anom_filePath.c_str(), ofstream::out);
    for(int i = 0; i < testNum; i++)
        anom_writeFile << anomScore[i] << " " << int(attack[i]) << endl;
    anom_writeFile.close();

    // COMPUTE AND WRITE ACCURACY
    string pr_filePath = base_name + "_precision_recall.txt";
    ofstream pr_writeFile;
    pr_writeFile.open(pr_filePath.c_str(), ofstream::out);
    for(int i = 1; i < 17; i ++)
    {
        int topN = 50 * i;
        AccuracyResult result = compute_accuracy(anomScore, attack, testNum, topN);

        // Use a stringstream with default formatting to match the old style
        stringstream ss;
        ss << "[TOP" << topN << "] precision: " << result.precision 
           << ", recall: " << result.recall;

        string output_line = ss.str();
        
        cout << output_line << endl;
        pr_writeFile << output_line << endl;
    }
    pr_writeFile.close();

    // FREE MEMORY
    delete [] A;
    delete [] anomScore;
    delete [] attack;

    for(int i = 0; i < 3; i++)
    {
       delete [] pagerank1[i];
       delete [] pagerank2[i];
    }
    delete [] pagerank1;
    delete [] pagerank2;

    for(int i = 0; i < 4; i++)
    {
        delete [] mean[i];
        delete [] var[i];
    }
    delete [] mean;
    delete [] var;

    return 1;
}