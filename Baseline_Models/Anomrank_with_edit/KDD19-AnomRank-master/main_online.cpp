#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <iomanip>

#include "edge.hpp"
#include "pagerank.cpp"
#include "anomaly_detect.cpp"
#include "anomaly_inject.cpp"

#define attackLimit 50

using namespace std;

/**
 Parameters:
 *   binIdx       : index of the current bin 
 *   binAttackNum : number of malicious edges in this bin
 *   hasEdge      : whether this bin had at least one edge
 *   n            : number of nodes in the graph
 *   A            : adjacency list / out-edge structure
 *   current_m    : total accumulated edge weight up to now
 *   initSS       : number of initial bins used as "training period"
 *   pagerank1/2  : PageRank score buffers (3 time slices x n nodes)
 *   mean, var    : statistics buffers used for anomaly normalization
 *   scores       : output vector for anomaly scores (for test bins)
 *   labels       : output vector for attack labels (0/1, for test bins)
 */
void finish_bin(
    int binIdx,
    int binAttackNum,
    bool hasEdge,
    int n,
    outEdge* A,
    int& current_m,
    int initSS,
    double** pagerank1,
    double** pagerank2,
    double** mean,
    double** var,
    std::vector<double>& scores,
    std::vector<int>& labels
) {
    // We only keep 3 time slices in memory and rotate them using modulo 3
    int iCur  =  binIdx      % 3;
    int iPrev = (binIdx > 0 ? binIdx - 1 : 0) % 3;

    // "Effective idle" means: no edges in this bin AND not the very first bin
    bool effective_idle = !hasEdge && (binIdx > 0);

    if (effective_idle) {
        // Idle bin: simply carry over the previous PageRank scores
        std::copy(pagerank1[iPrev], pagerank1[iPrev] + n, pagerank1[iCur]);
        std::copy(pagerank2[iPrev], pagerank2[iPrev] + n, pagerank2[iCur]);
    } else {
        // Active bin (or first bin): recompute PageRank for both directions
        pagerank(A, pagerank1[iCur], n, current_m, 1);
        pagerank(A, pagerank2[iCur], n, current_m, 2);
    }

    double score = 0.0;

    // During the training period (binIdx < initSS), we do not use the score
    if (binIdx >= initSS) {
        // compute_anomaly_score uses pagerank1/2 with time index binIdx
        score = compute_anomaly_score(binIdx, pagerank1, pagerank2, mean, var, n);
    }

    // Only store scores and labels after the training period
    if (binIdx >= initSS) {
        scores.push_back(score);

        // A bin is labeled as "attack" if it has at least attackLimit malicious edges
        int isAttack = (binAttackNum >= attackLimit) ? 1 : 0;
        labels.push_back(isAttack);
    }
}

int main(int argc, char* argv[])
{
    /**
     * Command-line arguments:
     *   argv[1] : stepSize  (time width of each bin, e.g., 60 seconds)
     *   argv[2] : initSS    (number of initial bins used for training)
     *   argv[3] : numNodes  (number of nodes in the graph, node IDs in [0, n-1])
     *
     * Input format (from stdin):
     *   t src trg [atk]
     *   - t   : integer timestamp (non-decreasing over time)
     *   - src : integer source node ID
     *   - trg : integer target node ID
     *   - atk : (optional) 0 or 1, indicating whether this edge is malicious
     *
     * Output format (to stdout):
     *   For each test bin (binIdx >= initSS), one line:
     *       <anomaly_score> <attack_label>
     *   where:
     *       anomaly_score : double (formatted with 6 decimal places)
     *       attack_label  : integer 0 or 1
     */

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <stepSize> <initSS> <numNodes>\n";
        std::cerr << "Example: " << argv[0] << " 60 100 10000 < edges.txt\n";
        return 1;
    }

    int stepSize = std::atoi(argv[1]);  // bin width in time units
    int initSS   = std::atoi(argv[2]);  // number of training bins
    int n        = std::atoi(argv[3]);  // number of nodes

    // Adjacency structure: one outEdge object per node
    outEdge* A = new outEdge[n];

    // Total number of edges (or total edge weight) seen so far
    int current_m = 0;

    // PageRank buffers: 3 time slices x n nodes
    double** pagerank1 = new double*[3];
    double** pagerank2 = new double*[3];
    for (int i = 0; i < 3; ++i) {
        pagerank1[i] = new double[n];
        pagerank2[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            pagerank1[i][j] = 0.0;
            pagerank2[i][j] = 0.0;
        }
    }

    // Online mean and variance buffers used by the anomaly detector
    // Dimensions: [4][n], depending on how compute_anomaly_score/normalize use them
    double** mean = new double*[4];
    double** var  = new double*[4];
    for (int i = 0; i < 4; ++i) {
        mean[i] = new double[n];
        var[i]  = new double[n];
        for (int j = 0; j < n; ++j) {
            mean[i][j] = 0.0;
            var[i][j]  = 0.0;
        }
    }

    // Output containers for anomaly scores and attack labels per bin
    std::vector<double> scores;
    std::vector<int>    labels;

    // Streaming state for time and bin management
    long long t0      = -1;   // timestamp of the first edge (bin reference)
    int       curBin  = 0;    // index of the current bin we are filling
    int       curAttackNum = 0;  // number of malicious edges in the current bin
    bool      curHasEdge   = false; // whether current bin has at least one edge

    // Helper lambda to close the current bin and reset its counters
    auto close_bin = [&](int binIdx) {
        finish_bin(
            binIdx,
            curAttackNum,
            curHasEdge,
            n,
            A,
            current_m,
            initSS,
            pagerank1,
            pagerank2,
            mean,
            var,
            scores,
            labels
        );
        // Reset per-bin statistics for the next bin
        curAttackNum = 0;
        curHasEdge   = false;
    };

    std::string line;
    // === Main streaming loop: read edges line by line from stdin ===
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        long long t;
        int src, trg;
        int atkFlag = 0;

        // Expected: t src trg [atk]
        if (!(iss >> t >> src >> trg)) {
            // If the line cannot be parsed, skip it
            continue;
        }
        // If attack flag is missing, default to 0 (benign)
        if (!(iss >> atkFlag)) {
            atkFlag = 0;
        }

        // Initialize the reference timestamp using the first edge
        if (t0 < 0) {
            t0 = t;
            curBin = 0;
        }

        // Compute how far this edge is from the first timestamp
        long long dt = t - t0;
        if (dt < 0) {
            // If timestamps are not strictly non-decreasing,
            // we clamp dt to 0 to avoid negative bin index
            dt = 0;
        }

        // Determine which bin this edge belongs to
        int binOfEdge = static_cast<int>(dt / stepSize);

        // If this edge belongs to a future bin,
        // we must close all intermediate bins first.
        while (binOfEdge > curBin) {
            close_bin(curBin);
            curBin++;
        }

        // Now binOfEdge == curBin, so we add the edge to the current bin
        if (src >= 0 && src < n && trg >= 0 && trg < n) {
            // Inject edge into the cumulative graph
            inject(A, src, trg, 1);
            current_m++;
        }
        if (atkFlag) curAttackNum++;
        curHasEdge = true;
    }

    // After the input stream ends, close the last bin as well
    close_bin(curBin);

    // === Output anomaly scores and labels for all test bins ===
    for (size_t i = 0; i < scores.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6)
                  << scores[i] << " " << labels[i] << "\n";
    }

    // === Clean up dynamically allocated memory ===
    delete [] A;
    for (int i = 0; i < 3; ++i) {
        delete [] pagerank1[i];
        delete [] pagerank2[i];
    }
    delete [] pagerank1;
    delete [] pagerank2;

    for (int i = 0; i < 4; ++i) {
        delete [] mean[i];
        delete [] var[i];
    }
    delete [] mean;
    delete [] var;

    return 0;
}
