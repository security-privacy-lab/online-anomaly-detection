#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

struct AccuracyResult {
    double precision;
    double recall;
    double f1;
    double tpr;
    double fpr;
};

AccuracyResult compute_accuracy(double* as, bool* attack, int timeNum, int topN)
{
    if (timeNum <= 0) {
        return {0.0, 0.0, 0.0, 0.0, 0.0};
    }
    if (topN < 0) topN = 0;
    if (topN > timeNum) topN = timeNum;

    std::vector<int> idx(timeNum);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(),
              [&](int i1, int i2){ return as[i1] > as[i2]; });

    int tp = 0, fp = 0, tn = 0, fn = 0;

    int positives = 0, negatives = 0;
    for (int i = 0; i < timeNum; ++i) {
        if (attack[i]) positives++; else negatives++;
    }

    for (int i = 0; i < topN; ++i) {
        if (attack[idx[i]]) tp++;
        else                fp++;
    }
    for (int i = topN; i < timeNum; ++i) {
        if (attack[idx[i]]) fn++;
        else                tn++;
    }

    double precision = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
    double recall    = (tp + fn) ? (double)tp / (tp + fn) : 0.0;
    double f1        = (precision + recall) ? (2.0 * precision * recall) / (precision + recall) : 0.0;
    double tpr       = recall;
    double fpr       = (fp + tn) ? (double)fp / (fp + tn) : 0.0;

    std::cout << "[TOP" << topN << "] precision: " << precision
              << ", recall: " << recall << std::endl;

    return {precision, recall, f1, tpr, fpr};
}
