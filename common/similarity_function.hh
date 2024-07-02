#pragma once

#include <vector>

extern double (*similarity_function)(const std::vector<double> &, const std::vector<double> &);

void functionSet();

double dotProduct(const std::vector<double> &v1, const std::vector<double> &v2);

double cosineSimilarity(const std::vector<double> &v1, const std::vector<double> &v2);

double hammingDistance(const std::vector<double> &v1, const std::vector<double> &v2);