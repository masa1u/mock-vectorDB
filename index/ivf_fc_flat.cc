#include "ivf_fc_flat.hh"

void ivf_fc_flat_worker(
    int thread_id,
    int &ready,
    const bool &start,
    const bool &quit,
    FuzzyCMeansIndex *index,
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

FuzzyCMeansIndex::FuzzyCMeansIndex(int num_clusters, int dimension, double fuzziness, double threshold)
    : num_clusters(num_clusters), dimension(dimension), fuzziness(fuzziness), threshold(threshold) {}

void FuzzyCMeansIndex::buildIndex(const std::vector<Vector *> &data)
{
  // 初期化
  centroids.resize(num_clusters);
  membership.resize(data.size(), std::vector<double>(num_clusters, 0.0));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<> gamma(1.0); // ディリクレ分布のパラメータα=1.0でガンマ分布を使用

  for (int i = 0; i < num_clusters; ++i)
  {
    centroids[i] = *data[std::uniform_int_distribution<>(0, data.size() - 1)(gen)];
  }

  // membershipをディリクレ分布で初期化し、合計が1になるように正規化
  for (auto &member : membership)
  {
    double sum = 0.0;
    std::vector<double> samples(num_clusters);
    for (double &sample : samples)
    {
      sample = gamma(gen); // ガンマ分布からサンプルを抽出
      sum += sample;
    }
    for (int i = 0; i < num_clusters; ++i)
    {
      member[i] = samples[i] / sum; // 合計が1になるように正規化
    }
  }

  bool changed;
  do
  {
    // クラスタ中心の計算
    calculateCentroids(data);

    // 所属度の計算
    std::vector<std::vector<double>> old_membership = membership;
    calculateMembership(data);

    // 所属度の変化をチェック
    changed = false;
    for (int i = 0; i < data.size(); ++i)
    {
      for (int j = 0; j < num_clusters; ++j)
      {
        if (fabs(membership[i][j] - old_membership[i][j]) > 1e-4)
        // if (membership[i][j] != old_membership[i][j])
        {
          changed = true;
          break;
        }
      }
      if (changed)
        break;
    }
  } while (changed);

  // 所属度が閾値より大きいクラスタにベクトルを格納
  clusters.resize(num_clusters);
  for (int i = 0; i < data.size(); ++i)
  {
    for (int j = 0; j < num_clusters; ++j)
    {
      if (membership[i][j] > threshold)
      {
        clusters[j].push_back(*data[i]);
      }
    }
  }
}

std::vector<int> FuzzyCMeansIndex::search(const Vector &query, int top_k, int n_probe)
{
  std::vector<std::pair<int, double>> distances;
  for (int i = 0; i < n_probe; ++i)
  {
    int cluster_id = nthClosestCentroid(query, i + 1);

    for (int j = 0; j < clusters[cluster_id].size(); ++j)
    {
      double dist = similarity_function(query.features, clusters[cluster_id][j].features);
      distances.emplace_back(clusters[cluster_id][j].id, dist);
    }
  }

  std::sort(distances.begin(), distances.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b)
            { return a.second < b.second; });

  std::set<int> uniqueIds;
  std::vector<int> result;
  for (int i = 0; i < distances.size() && uniqueIds.size() < top_k; ++i)
  {
    // 重複をチェックし、重複がなければ結果に追加
    if (uniqueIds.insert(distances[i].first).second)
    {
      result.push_back(distances[i].first);
    }
  }
  return result;
}

void FuzzyCMeansIndex::calculateCentroids(const std::vector<Vector *> &data)
{
  for (int j = 0; j < num_clusters; ++j)
  {
    Vector new_centroid(std::vector<double>(dimension, 0.0));
    double total_weight = 0.0;
    for (int i = 0; i < data.size(); ++i)
    {
      double weight = pow(membership[i][j], fuzziness);
      for (int k = 0; k < dimension; ++k)
      {
        new_centroid.features[k] += data[i]->features[k] * weight;
      }
      total_weight += weight;
    }
    for (int k = 0; k < dimension; ++k)
    {
      new_centroid.features[k] /= total_weight;
    }
    centroids[j] = new_centroid;
  }
}

void FuzzyCMeansIndex::calculateMembership(const std::vector<Vector *> &data)
{
  for (int i = 0; i < data.size(); ++i)
  {
    // double sum = 0.0;
    for (int j = 0; j < num_clusters; ++j)
    {
      double sum = 0.0;
      for (int k = 0; k < num_clusters; ++k)
      {
        sum += pow(similarity_function(data[i]->features, centroids[j].features) / similarity_function(data[i]->features, centroids[k].features), 2.0 / (fuzziness - 1.0));
      }
      membership[i][j] = 1 / sum;
      // sum += membership[i][j];
    }
    // // 正規化
    // for (int j = 0; j < num_clusters; ++j)
    // {
    //   membership[i][j] /= sum;
    // }
  }
}

int FuzzyCMeansIndex::nthClosestCentroid(const Vector &point, int n)
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

void FuzzyCMeansIndex::printClusters() const
{
  for (int i = 0; i < num_clusters; ++i)
  {
    std::cout << "[IVF-FC]Cluster " << i << ":\n";
    for (const auto &vector : clusters[i])
    {
      // ベクトルの特徴ではなく、ベクトルのIDを出力
      std::cout << "Vector ID: " << vector.id << "\n";
    }
    std::cout << "\n";
  }
}

void FuzzyCMeansIndex::printMembership() const
{
  for (int i = 0; i < membership.size(); ++i)
  {
    std::cout << "[IVF-FC]Vector " << i << ":\n";
    for (int j = 0; j < num_clusters; ++j)
    {
      std::cout << "Cluster " << j << ": " << membership[i][j] << "\n";
    }
    std::cout << "\n";
  }
}

void FuzzyCMeansIndex::printCentroids() const
{
  for (int i = 0; i < num_clusters; ++i)
  {
    std::cout << "[IVF-FC]Centroid " << i << ":\n";
    for (int j = 0; j < dimension; ++j)
    {
      std::cout << centroids[i].features[j] << " ";
    }
    std::cout << "\n";
  }
}