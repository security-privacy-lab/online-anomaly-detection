#include <vector>
#include <iostream>
#include <numeric>      // Required for std::iota
#include <algorithm>    // Required for std::sort
#include <ostream>      // Required for std::ostream

// A struct to hold the results of our calculation
struct AccuracyResult
{
    double precision;
    double recall;
};

AccuracyResult compute_accuracy(double* as, bool* attack, int timeNum, int topN)
{
    int tp, fp, tn, fn;
    tp = fp = tn = fn = 0;
    std::vector<size_t> idx(timeNum);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&as](size_t i1, size_t i2) {return as[i1] > as[i2];});

    for(int i = 0; i < topN; i++)
    {
        if(attack[idx[i]])
            tp++;
        else
            fp++;
	}

    for(int i = topN; i < timeNum; i++)
    {
        if(attack[idx[i]])
            fn++;
        else
            tn++;
    }

    AccuracyResult result;
    result.precision = (tp + fp == 0) ? 0 : double(tp)/(tp+fp);
    result.recall = (tp + fn == 0) ? 0 : double(tp)/(tp+fn);
    
    // Return the result object
    return result;
}