#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

void compute_accuracy(double* as, bool* attack, int timeNum, int topN, double &precision, double &recall) {
    std::vector<int> idx(timeNum);
    std::iota(idx.begin(), idx.end(), 0);

    // Sort indices based on anomaly scores
    std::sort(idx.begin(), idx.end(), [as](int i1, int i2) {
        return as[i1] > as[i2];
    });

    int truePositives = 0;
    int falsePositives = 0;
    int actualAnomalies = 0;

    // Count actual anomalies in the data
    for (int i = 0; i < timeNum; ++i) {
        if (attack[i]) {
            actualAnomalies++;
        }
    }

    // Calculate true positives and false positives for topN results
    for (int i = 0; i < topN; ++i) {
        if (attack[idx[i]]) {
            truePositives++;
        } else {
            falsePositives++;
        }
    }

    precision = (topN > 0) ? static_cast<double>(truePositives) / topN : 0.0;
    recall = (actualAnomalies > 0) ? static_cast<double>(truePositives) / actualAnomalies : 0.0;

    // Commented out to prevent duplicate output:
    // std::cout << "Top " << topN << " - Precision: " << precision << ", Recall: " << recall << std::endl;
}
