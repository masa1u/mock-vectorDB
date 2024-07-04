#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <cfloat>

#include "../common/dataset.hh"
#include "../common/similarity_function.hh"

// IVF-FLATインデックスクラス
class IVFFlatIndex
{
public:
  IVFFlatIndex(int num_clusters, int dimension);
  void buildIndex(const std::vector<Vector *> data);
  std::vector<int> search(const Vector &query, int top_k);

private:
  int num_clusters;
  int dimension;
  std::vector<Vector> centroids;
  std::vector<std::vector<Vector>> clusters;

  void kmeans(const std::vector<Vector *> data);
  int closestCentroid(const Vector &point);
};