#ifndef EDGE_H
#define EDGE_H

#include <vector>

struct timeEdge {
    int orig_t;
    int orig_src;
    int orig_trg;
    unsigned char atk;

    int t;
    int src;
    int trg;

    timeEdge() : orig_t(0), orig_src(0), orig_trg(0), atk(0), t(0), src(0), trg(0) {}

    explicit timeEdge(std::vector<int>& tokens) {
        orig_t   = tokens[0];
        orig_src = tokens[1];
        orig_trg = tokens[2];
        atk      = (tokens.size() >= 4 ? (tokens[3] == 1) : 0);
        t = orig_t; src = orig_src; trg = orig_trg;
    }
};

struct outEdge {
    outEdge() : total_w(0.0) {}
    double total_w;
    std::vector<int> out;
    std::vector<int> weight;
};

#endif
