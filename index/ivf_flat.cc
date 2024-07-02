#include "ivf_flat.hh"
#include "../common/similarity_function.hh"
#include "../common/dataset.hh"

// コンストラクタ
IVFFlatIndex::IVFFlatIndex(int num_clusters, int dimension)
    : num_clusters(num_clusters), dimension(dimension) {}

// インデックスの構築
void IVFFlatIndex::buildIndex(const std::vector<Vector *> data)
{
  // k-means法を用いてクラスタ中心を求める
  kmeans(data);

  // クラスタリングされたデータを格納するためのベクトルを初期化
  clusters.resize(num_clusters);
  for (const auto &point : data)
  {
    int cluster_id = closestCentroid(point->features);
    clusters[cluster_id].push_back(point->features);
  }
}

void IVFFlatIndex::kmeans(const std::vector<Vector *> data)
{
  // 初期クラスタ中心をランダムに選択
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, data.size() - 1);

  centroids.resize(num_clusters);
  for (int i = 0; i < num_clusters; ++i)
  {
    centroids[i] = data[dis(gen)]->features;
  }

  bool changed;
  do
  {
    // 各データポイントを最も近いクラスタ中心に割り当てる
    std::vector<std::vector<Vector>> new_clusters(num_clusters);
    for (const auto &point : data)
    {
      int cluster_id = closestCentroid(point->features);
      new_clusters[cluster_id].push_back(point->features);
    }

    // 新しいクラスタ中心を計算
    changed = false;
    for (int i = 0; i < num_clusters; ++i)
    {
      if (new_clusters[i].empty())
        continue;

      Vector new_centroid(std::vector<double>(dimension, 0.0));

      for (const auto &point : new_clusters[i])
      {
        for (int j = 0; j < dimension; ++j)
        {
          new_centroid.features[j] += point.features[j];
        }
      }
      for (int j = 0; j < dimension; ++j)
      {
        new_centroid.features[j] /= new_clusters[i].size();
      }

      if (similarity_function(centroids[i].features, new_centroid.features) > 1e-4)
      {
        changed = true;
        centroids[i] = new_centroid;
      }
    }
  } while (changed);
}

int IVFFlatIndex::closestCentroid(const Vector &point)
{
  int closest = -1;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < num_clusters; ++i)
  {
    double dist = similarity_function(point.features, centroids[i].features);
    if (dist < min_dist)
    {
      min_dist = dist;
      closest = i;
    }
  }
  return closest;
}

std::vector<std::vector<double>> IVFFlatIndex::search(const Vector &query, int top_k)
{
  // クエリに最も近いクラスタ中心を見つける
  int cluster_id = closestCentroid(query);

  // クラスタ内の全データポイントに対して距離を計算
  std::vector<std::pair<double, std::vector<double>>> distances;
  for (int i = 0; i < clusters[cluster_id].size(); ++i)
  {
    double dist = similarity_function(query.features, clusters[cluster_id][i].features);
    distances.emplace_back(dist, clusters[cluster_id][i].features);
  }

  // 距離でソートして上位top_kを返す
  std::sort(distances.begin(), distances.end());
  std::vector<std::vector<double>> result;
  for (int i = 0; i < std::min(top_k, static_cast<int>(distances.size())); ++i)
  {
    result.push_back(distances[i].second);
  }
  return result;
}