#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>
#include <fstream>
#include <iostream>
#include <thread>

#include "../common/dataset.hh"
#include "../common/similarity_function.hh"
#include "../common/result.hh"

class FuzzyCMeansIndex
{
public:
  FuzzyCMeansIndex(int num_clusters, int dimension, double fuzziness, double threshold);
  void buildIndex(const std::vector<Vector *> &data);
  std::vector<int> search(const Vector &query, int top_k, int n_probe);
  void saveIndex(const std::string &filename) const;
  void loadIndex(const std::string &filename);
  void printClusters() const;   // debug
  void printMembership() const; // debug
  void printCentroids() const;  // debug

private:
  int num_clusters;
  int dimension;
  double fuzziness; // ファジィ度
  double threshold; // クラスタ所属判定の閾値
  std::vector<Vector> centroids;
  std::vector<std::vector<double>> membership; // 各ベクトルのクラスタへの所属度
  std::vector<std::vector<Vector>> clusters;

  void calculateCentroids(const std::vector<Vector *> &data);
  void calculateMembership(const std::vector<Vector *> &data);
  int nthClosestCentroid(const Vector &point, int n);
  void calculateMembershipForDataPoint(const std::vector<Vector *> &data, int i);
};

void ivf_fc_flat_worker(int thread_id, int &ready, const bool &start, const bool &quit, FuzzyCMeansIndex *index, int dimension, int top_k, int nprobe);