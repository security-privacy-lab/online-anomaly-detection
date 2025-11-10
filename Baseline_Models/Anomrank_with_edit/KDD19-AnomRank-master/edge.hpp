#ifndef EDGE_H
#define EDGE_H

#include <vector>

struct timeEdge {
    // 원본(입력) 보존
    int orig_t;
    int orig_src;
    int orig_trg;
    unsigned char atk;  // 0 or 1

    // 내부(정규화/압축) 값
    int t;   // orig_t - t0
    int src; // orig_src -> 0..n-1
    int trg; // orig_trg -> 0..n-1

    timeEdge() : orig_t(0), orig_src(0), orig_trg(0), atk(0), t(0), src(0), trg(0) {}

    // tokens = [t, src, trg, (opt) atk]
    explicit timeEdge(std::vector<int>& tokens) {
        orig_t   = tokens[0];
        orig_src = tokens[1];
        orig_trg = tokens[2];
        atk      = (tokens.size() >= 4 ? (tokens[3] == 1) : 0);
        t = orig_t; src = orig_src; trg = orig_trg; // read_data에서 정규화
    }
};

struct outEdge {
    outEdge() : total_w(0.0) {}
    double total_w;
    std::vector<int> out;
    std::vector<int> weight;
};

#endif
