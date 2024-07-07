#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <cfloat>

#include "../common/dataset.hh"
#include "../common/similarity_function.hh"
#include "../common/result.hh"

// IVF-FLATインデックスクラス
class IVFFlatIndex
{
public:
  IVFFlatIndex(int num_clusters, int dimension);
  void buildIndex(const std::vector<Vector *> data);
  std::vector<int> search(const Vector &query, int top_k, int n_probe);

private:
  int num_clusters;
  int dimension;
  std::vector<Vector> centroids;
  std::vector<std::vector<Vector>> clusters;

  void kmeans(const std::vector<Vector *> data);
  int nthClosestCentroid(const Vector &point, int n);
};

void ivf_flat_worker(int thread_id, int &ready, const bool &start, const bool &quit, IVFFlatIndex *index, int dimension, int top_k, int nprobe);