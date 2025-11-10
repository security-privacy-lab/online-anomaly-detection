#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "edge.hpp"
using namespace std;

static inline void split_whitespace(const string& line, vector<int>& out) {
    out.clear();
    istringstream iss(line);
    int x; while (iss >> x) out.push_back(x);
}

void read_data(string path, string /*delimiter_ignored*/, int stepSize,
               vector<timeEdge>& edges, vector<int>& snapshots,
               int& n, int& m, int& timeNum)
{
    edges.clear(); snapshots.clear();
    vector<int> all_nodes, all_times;

    ifstream fin(path.c_str());
    if (!fin) { cerr << "Cannot open: " << path << "\n"; n=m=timeNum=0; return; }

    string line; vector<int> tokens;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        split_whitespace(line, tokens);
        if (tokens.size() < 3) continue;
        edges.emplace_back(tokens);
        all_times.push_back(tokens[0]);
        all_nodes.push_back(tokens[1]);
        all_nodes.push_back(tokens[2]);
    }
    fin.close();
    if (edges.empty()) { n=m=timeNum=0; return; }

    // 같은 t 내 입력 순서 유지
    stable_sort(edges.begin(), edges.end(),
        [](const timeEdge& a, const timeEdge& b){ return a.orig_t < b.orig_t; });

    // 노드 압축
    sort(all_nodes.begin(), all_nodes.end());
    all_nodes.erase(unique(all_nodes.begin(), all_nodes.end()), all_nodes.end());
    unordered_map<int,int> idmap; idmap.reserve(all_nodes.size()*2);
    for (size_t i=0;i<all_nodes.size();++i) idmap[ all_nodes[i] ] = (int)i;

    // 시간 0 기준 정규화
    sort(all_times.begin(), all_times.end());
    all_times.erase(unique(all_times.begin(), all_times.end()), all_times.end());
    const int t0   = all_times.front();
    const int tmax = all_times.back();

    for (auto& e : edges) {
        e.t   = e.orig_t - t0;
        e.src = idmap[e.orig_src];
        e.trg = idmap[e.orig_trg];
    }

    n = (int)all_nodes.size();
    m = (int)edges.size();
    timeNum = (tmax - t0 + 1);

    // ---- 연속 bin + prefix snapshots ----
    if (stepSize <= 0) { cerr << "Invalid stepSize\n"; return; }
    const int numBins = (edges.back().t / stepSize) + 1;

    vector<int> cnt(numBins, 0), ebin(m, 0);
    for (int i = 0; i < m; ++i) {
        int b = edges[i].t / stepSize;
        if (b < 0) b = 0;
        if (b >= numBins) b = numBins - 1;
        ebin[i] = b;
        cnt[b]++;
    }

    snapshots.assign(numBins + 1, 0);
    for (int b = 0; b < numBins; ++b) snapshots[b+1] = snapshots[b] + cnt[b];

    vector<timeEdge> reordered(m);
    vector<int> writePos = snapshots;
    for (int i = 0; i < m; ++i) {
        int b = ebin[i];
        reordered[ writePos[b]++ ] = edges[i];
    }
    edges.swap(reordered);
    // 이제 각 bin의 엣지 범위는 [snapshots[b], snapshots[b+1])
}
