#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>

#include "accuracy.cpp"
#include "anomaly_detect.cpp"
#include "anomaly_inject.cpp"
#include "pagerank.cpp"
#include "read_data.cpp"

#define attackLimit 50

using namespace std;

int main(int argc, char *argv[])
{
    clock_t start = clock();

    
    if (argc < 8) {
        cerr << "Usage: " << argv[0]
             << " <path> <delimiter> <timeStep> <initSS> <injectScene> <injectNum> <injectSize>\n";
        return 1;
    }

    string path       = argv[1];
    string delimeter  = argv[2];  (void)delimeter; 
    int timeStep      = atoi(argv[3]);
    int initSS        = atoi(argv[4]);
    int injectScene   = atoi(argv[5]);
    int injectNum     = atoi(argv[6]);
    int injectSize    = atoi(argv[7]);
    bool INJECT       = (injectScene != 0);

 
    vector<timeEdge> edges;
    vector<int> snapshots;
    int n=0, m=0, timeNum=0;
    read_data(path, delimeter, timeStep, edges, snapshots, n, m, timeNum);

    int numBins = (int)snapshots.size() - 1;        
    outEdge* A = new outEdge[n]();
    cout << "#node: " << n << ", #edges: " << edges.size()
         << ", #timeStamp: " << timeNum << endl;

  
    int testNum = numBins - initSS;
    if (testNum < 0) testNum = 0;

    bool*   attack    = new bool  [max(1, testNum)];
    double* anomScore = new double[max(1, testNum)];
    for (int i = 0; i < testNum; i++) {
        anomScore[i] = 0.0;
        attack[i]    = false;
    }


    vector<double> edgeScore(edges.size(), 0.0);
    vector<unsigned char> edgeLabel(edges.size(), 0);


    double** pagerank1 = new double*[3];
    double** pagerank2 = new double*[3];
    for (int i = 0; i < 3; i++) {
        pagerank1[i] = new double[n];
        pagerank2[i] = new double[n];
        for (int j = 0; j < n; j++)
            pagerank1[i][j] = pagerank2[i][j] = 0.0;
    }

    double** mean = new double*[4];
    double** var  = new double*[4];
    for (int i = 0; i < 4; i++) {
        mean[i] = new double[n];
        var[i]  = new double[n];
        for (int j = 0; j < n; j++)
            mean[i][j] = var[i][j] = 0.0;
    }


    vector<int> injectSS;
    if (INJECT && injectNum > 0) {
        int usable  = max(0, numBins - initSS);
        int gap     = max(1, usable / injectNum);
        for (int k = 0; k < injectNum; ++k) {
            int b = initSS + k * gap;
            if (0 <= b && b < numBins) injectSS.push_back(b);
        }
        sort(injectSS.begin(), injectSS.end());
        injectSS.erase(unique(injectSS.begin(), injectSS.end()), injectSS.end());
    }

    cout << "Preprocess done: " << (double)(clock() - start) / CLOCKS_PER_SEC << endl;

    int current_m = 0;            
    int injected  = 0;         
    start = clock();
    int print_e = 10;

    for (int b = 0; b < numBins; ++b) {
        const int start_eg = snapshots[b];
        const int end_eg   = snapshots[b+1];   
        const bool idle    = (start_eg == end_eg);
        int binAttackNum = 0;  

        for (int i = start_eg; i < end_eg; ++i) {
            inject(A, edges[i].src, edges[i].trg, 1);
            if (edges[i].atk) binAttackNum++;
            current_m++;
            if (i + 1 == print_e) {
                cout << (i + 1) << "," << (double)(clock() - start) / CLOCKS_PER_SEC << endl;
                print_e *= 10;
            }
        }

       
        bool injected_this_bin = false;
        if (INJECT && injected < (int)injectSS.size() && injectSS[injected] == b) {
            current_m += inject_anomaly(injectScene, A, n, injectSize);
            injected++;
            injected_this_bin = true;
            if (injected == (int)injectSS.size()) INJECT = false;
        }

        const bool effective_idle = (start_eg == end_eg) && !injected_this_bin;

        
        const int iCur  =  b      % 3;
        const int iPrev = (b ? b-1 : b) % 3;
        if (effective_idle && b > 0) {
            std::copy(pagerank1[iPrev], pagerank1[iPrev] + n, pagerank1[iCur]);
            std::copy(pagerank2[iPrev], pagerank2[iPrev] + n, pagerank2[iCur]);
        } else {
            pagerank(A, pagerank1[iCur], n, current_m, 1);
            pagerank(A, pagerank2[iCur], n, current_m, 2);
        }

   
        double score = (b < initSS)
            ? 0.0
            : compute_anomaly_score(b, pagerank1, pagerank2, mean, var, n);
   
        if (b >= initSS) {
            const int idx = b - initSS;
            if (idx >= 0 && idx < testNum) {
                anomScore[idx] = score;
           
                attack[idx]    = (binAttackNum >= attackLimit);
            }
        }

      
        {
            const unsigned char isAtk = (binAttackNum >= attackLimit) ? 1 : 0;
            for (int i = start_eg; i < end_eg; ++i) {
                edgeScore[i] = score;
                edgeLabel[i] = isAtk;
            }
        }
    }


 
    string base_name = path;
    size_t last_slash = base_name.find_last_of("/\\");
    if (last_slash != string::npos) base_name.erase(0, last_slash + 1);
    size_t extension_pos = base_name.find_last_of(".");
    if (extension_pos != string::npos) base_name.erase(extension_pos);

    {
        string anom_filePath = base_name + "_anomaly.txt";
        ofstream fout(anom_filePath.c_str());
        for (int i = 0; i < testNum; ++i)
            fout << anomScore[i] << " " << int(attack[i]) << "\n";
        fout.close();
        cout << "Wrote: " << anom_filePath << endl;
    }

  
    {
        vector<int> order(testNum);
        for (int i = 0; i < testNum; ++i) order[i] = i;
        sort(order.begin(), order.end(), [&](int a, int b){ return anomScore[a] > anomScore[b]; });

        int K = min(10, testNum);
        string top_filePath = base_name + "_top.txt";
        ofstream tout(top_filePath.c_str());
        tout << "#bin_index score label\n";
        for (int k = 0; k < K; ++k) {
            int idx = order[k];
            int bin = idx + initSS;
            tout << bin << " " << anomScore[idx] << " " << int(attack[idx]) << "\n";
        }
        tout.close();
        cout << "Wrote: " << top_filePath << endl;
    }


    {
        string pr_filePath = base_name + "_precision_recall.txt";
        ofstream prout(pr_filePath.c_str());
        for (int i = 1; i < 17; ++i) {
            int topN = 50 * i;
            AccuracyResult result = compute_accuracy(anomScore, attack, testNum, topN);
            prout << "[TOP" << topN << "] precision: " << result.precision
                  << ", recall: " << result.recall << "\n";
        }
        prout.close();
        cout << "Wrote: " << pr_filePath << endl;
    }

  
    {
        string edge_filePath = base_name + "_edge_anomaly.txt";
        ofstream eout(edge_filePath.c_str());
        eout << "#t src trg atk score\n";
        eout << fixed << setprecision(6);
        for (size_t i = 0; i < edges.size(); ++i) {
            eout << edges[i].orig_t   << " "
                 << edges[i].orig_src << " "
                 << edges[i].orig_trg << " "
                 << int(edges[i].atk) << " "
                 << edgeScore[i]      << "\n";
        }
        eout.close();
        cout << "Wrote: " << edge_filePath << endl;
    }

    {
        string csv_path = base_name + "_edge_anomaly.csv";
        ofstream csv(csv_path.c_str());
        csv << "edge_index,t,src,trg,atk,score\n";
        csv << fixed << setprecision(6);
        for (size_t i = 0; i < edges.size(); ++i) {
            csv << i << ","
                << edges[i].orig_t   << ","
                << edges[i].orig_src << ","
                << edges[i].orig_trg << ","
                << int(edges[i].atk) << ","
                << edgeScore[i]      << "\n";
        }
        csv.close();
        cout << "Wrote: " << csv_path << endl;
    }

    {
        string xml_path = base_name + "_edge_anomaly.xml";
        ofstream xout(xml_path.c_str());
        xout << R"(<?xml version="1.0"?>)"
             << "\n" << R"(<?mso-application progid="Excel.Sheet"?>)"
             << "\n" << R"(<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
    xmlns:o="urn:schemas-microsoft-com:office:office"
    xmlns:x="urn:schemas-microsoft-com:office:excel"
    xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">)"
             << "\n"
             << R"(<Worksheet ss:Name="edge_anomaly"><Table>)" << "\n";

        xout << "<Row>"
             << "<Cell><Data ss:Type=\"String\">edge_index</Data></Cell>"
             << "<Cell><Data ss:Type=\"String\">t</Data></Cell>"
             << "<Cell><Data ss:Type=\"String\">src</Data></Cell>"
             << "<Cell><Data ss:Type=\"String\">trg</Data></Cell>"
             << "<Cell><Data ss:Type=\"String\">atk</Data></Cell>"
             << "<Cell><Data ss:Type=\"String\">score</Data></Cell>"
             << "</Row>\n";

        xout << fixed << setprecision(6);
        for (size_t i = 0; i < edges.size(); ++i) {
            xout << "<Row>";
            xout << "<Cell><Data ss:Type=\"Number\">" << i                 << "</Data></Cell>";
            xout << "<Cell><Data ss:Type=\"Number\">" << edges[i].orig_t   << "</Data></Cell>";
            xout << "<Cell><Data ss:Type=\"Number\">" << edges[i].orig_src << "</Data></Cell>";
            xout << "<Cell><Data ss:Type=\"Number\">" << edges[i].orig_trg << "</Data></Cell>";
            xout << "<Cell><Data ss:Type=\"Number\">" << int(edges[i].atk) << "</Data></Cell>";
            xout << "<Cell><Data ss:Type=\"Number\">" << edgeScore[i]      << "</Data></Cell>";
            xout << "</Row>\n";
        }

        xout << "</Table></Worksheet></Workbook>\n";
        xout.close();
        cout << "Wrote: " << xml_path << endl;
    }

    delete [] A;
    delete [] anomScore;
    delete [] attack;

    for (int i = 0; i < 3; i++) {
        delete [] pagerank1[i];
        delete [] pagerank2[i];
    }
    delete [] pagerank1;
    delete [] pagerank2;

    for (int i = 0; i < 4; i++) {
        delete [] mean[i];
        delete [] var[i];
    }
    delete [] mean;
    delete [] var;

    return 0;
}
