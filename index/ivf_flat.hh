#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <cfloat>

#include "../common/dataset.hh"

// // データポイントを表す型定義
// using DataPoint = std::vector<double>;

// IVF-FLATインデックスクラス
class IVFFlatIndex
{
public:
  IVFFlatIndex(int num_clusters, int dimension);
  void buildIndex(const std::vector<Vector *> data);
  std::vector<std::vector<double>> search(const Vector &query, int top_k);

private:
  int num_clusters;
  int dimension;
  std::vector<Vector> centroids;
  std::vector<std::vector<Vector>> clusters;

  void kmeans(const std::vector<Vector *> data);
  int closestCentroid(const Vector &point);
};