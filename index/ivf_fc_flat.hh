#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <set>

#include "../common/dataset.hh"
#include "../common/similarity_function.hh"
#include "../common/result.hh"

class FuzzyCMeansIndex
{
public:
  FuzzyCMeansIndex(int num_clusters, int dimension, double fuzziness);
  void buildIndex(const std::vector<Vector *> &data);
  std::vector<int> search(const Vector &query, int top_k, int n_probe);

private:
  int num_clusters;
  int dimension;
  double fuzziness; // ファジィ度
  std::vector<Vector> centroids;
  std::vector<std::vector<double>> membership; // 各ベクトルのクラスタへの所属度
  std::vector<std::vector<Vector>> clusters;

  void calculateCentroids(const std::vector<Vector *> &data);
  void calculateMembership(const std::vector<Vector *> &data);
  int nthClosestCentroid(const Vector &point, int n);
};

void ivf_fc_flat_worker(int thread_id, int &ready, const bool &start, const bool &quit, FuzzyCMeansIndex *index, int dimension, int top_k, int nprobe);