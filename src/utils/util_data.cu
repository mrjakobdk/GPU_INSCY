
#include <torch/extension.h>
#include "util_data.h"
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


using namespace std;

vector<vector<float>> load_glove(int n_max, int d_max) {

    fstream file;
    file.open("data/glove.6B.100d.txt");
    char seperator = ' ';

    string line;

    vector <vector<float>> m;
    bool first_line = true;
    int count = 0;
    while (getline(file, line, '\n')) {
        if (first_line) {
            first_line = false;
            continue;
        }
        istringstream templine(line);
        string data;

        bool first_col = true;
        vector<float> row;
        int d = 0;
        while (getline(templine, data, seperator) && d < d_max) {
            if (first_col) {
                first_col = false;
                continue;
            }
            row.push_back(atof(data.c_str()));
            d++;
        }
        m.push_back(row);
        count++;
        if (count >= n_max)
            break;
    }
    file.close();
    return m;
}


at::Tensor load_glove_torch(int n_max, int d_max) {

    fstream file;
    file.open("data/glove.6B.100d.txt");
    char seperator = ' ';

    string line;

    vector <vector<float>> m;
    bool first_line = true;
    int count = 0;
    while (getline(file, line, '\n')) {
        if (first_line) {
            first_line = false;
            continue;
        }
        istringstream templine(line);
        string data;

        bool first_col = true;
        vector<float> row;
        int d = 0;
        while (getline(templine, data, seperator) && d < d_max) {
            if (first_col) {
                first_col = false;
                continue;
            }
            row.push_back(atof(data.c_str()));
            d++;
        }
        m.push_back(row);
        count++;
        if (count >= n_max)
            break;
    }
    file.close();

    int n = m.size();
    int d =  m[0].size();

    at::Tensor X = at::zeros({n, d}, at::kFloat);
    for(int i =0;i<n;i++){
        for(int j =0;j<d;j++){
            X[i][j] = m[i][j];
        }
    }

    return X;
}

vector<vector<float>> load_gene(int n_max) {

    fstream file;
    file.open("data/data.csv");//https://www.kaggle.com/murats/gene-expression-cancer-rnaseq
    char seperator = ',';

    string line;

    vector <vector<float>> m;
    bool first_line = true;
    int count = 0;
    while (getline(file, line, '\n')) {
        if (first_line) {
            first_line = false;
            continue;
        }
        istringstream templine(line);
        string data;

        bool first_col = true;
        vector<float> row;
        while (getline(templine, data, seperator)) {
            if (first_col) {
                first_col = false;
                continue;
            }
            row.push_back(atof(data.c_str()));
        }
        m.push_back(row);
        count++;
        if (count >= n_max)
            break;
    }
    file.close();
    return m;
}

vector<vector<float>> load_glass(int n_max) {

    fstream file;
    file.open("data/glass/glass.data");
    char seperator = ',';

    string line;

    vector <vector<float>> m;
    int count = 0;
    while (getline(file, line, '\n')) {
        istringstream templine(line);
        string data;

        bool first_col = true;
        vector<float> row;
        while (getline(templine, data, seperator)) {
            if (first_col) {
                first_col = false;
                continue;
            }
            row.push_back(atof(data.c_str()));
        }
        row.pop_back();
        m.push_back(row);
        count++;
        if (count >= n_max)
            break;
    }
    file.close();
    return m;
}


vector<vector<float>> load_vowel(int n_max) {

    fstream file;
    file.open("data/real/vowel.dat");
    char seperator = ',';

    string line;

    vector <vector<float>> m;
    int count = 0;
    while (getline(file, line, '\n')) {
        istringstream templine(line);
        string data;

        bool first_col = true;
        vector<float> row;
        while (getline(templine, data, seperator)) {
            if (first_col) {
                first_col = false;
                continue;
            }
            row.push_back(atof(data.c_str()));
        }
        row.pop_back();
        m.push_back(row);
        count++;
        if (count >= n_max)
            break;
    }
    file.close();
    return m;
}