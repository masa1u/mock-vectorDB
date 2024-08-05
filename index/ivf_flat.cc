#include "ivf_flat.hh"

void ivf_flat_worker(
    int thread_id,
    int &ready,
    const bool &start,
    const bool &quit,
    IVFFlatIndex *index,
    int dimension,
    int top_k,
    int nprobe)
{
  Result &myres = std::ref(IndexResults[thread_id]);

  __atomic_store_n(&ready, 1, __ATOMIC_SEQ_CST);
  while (!__atomic_load_n(&start, __ATOMIC_SEQ_CST))
  {
  }

  while (!__atomic_load_n(&quit, __ATOMIC_SEQ_CST))
  {
    Vector *query_vector = createRandomVector(dimension);

    std::vector<int> result = index->search(*query_vector, top_k, nprobe);

    myres.search_results.emplace_back(*query_vector, result);
    myres.queries_count++;
  }
}

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
    int cluster_id = nthClosestCentroid(point->features, 1);
    clusters[cluster_id].push_back(*point);
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
      int cluster_id = nthClosestCentroid(point->features, 1);
      new_clusters[cluster_id].push_back(*point);
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

int IVFFlatIndex::nthClosestCentroid(const Vector &point, int n)
{
  std::vector<std::pair<double, int>> distances;
  for (int i = 0; i < num_clusters; ++i)
  {
    double dist = similarity_function(point.features, centroids[i].features);
    distances.emplace_back(dist, i);
  }
  std::sort(distances.begin(), distances.end());

  // nがクラスタの数より大きい場合や負の値の場合はエラー値を返す
  if (n < 1 || n > num_clusters)
    return -1;

  // 0-indexedなので、n-1番目の要素を返す
  return distances[n - 1].second;
}

std::vector<int> IVFFlatIndex::search(const Vector &query, int top_k, int n_probe)
{
  std::vector<std::pair<int, double>> distances;
  int current_cluster = 1;
  while (current_cluster <= n_probe or distances.size() < top_k)
  {
    int cluster_id = nthClosestCentroid(query, current_cluster);
    for (int j = 0; j < clusters[cluster_id].size(); ++j)
    {
      double dist = similarity_function(query.features, clusters[cluster_id][j].features);
      distances.emplace_back(clusters[cluster_id][j].id, dist);
    }
    current_cluster++;
  }

  // 距離でソートして上位top_kを返す
  std::sort(distances.begin(), distances.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b)
            { return a.second < b.second; });
  std::vector<int> result;
  for (int i = 0; i < std::min(top_k, static_cast<int>(distances.size())); ++i)
  {
    result.push_back(distances[i].first);
  }
  return result;
}

void IVFFlatIndex::printClusters() const
{
  for (int i = 0; i < num_clusters; ++i)
  {
    std::cout << "[IVF-FLAT]Cluster " << i << ":\n";
    for (const auto &vector : clusters[i])
    {
      // ベクトルの特徴ではなく、ベクトルのIDを出力
      std::cout << "Vector ID: " << vector.id << "\n";
    }
    std::cout << "\n";
  }
}