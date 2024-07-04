#include "flat.hh"

void FlatIndex::buildIndex(const std::vector<Vector *> dataset)
{
  data = dataset;
}

std::vector<int> FlatIndex::search(const Vector &query, int top_k)
{
  std::vector<std::pair<int, double>> distances;
  for (int i = 0; i < data.size(); i++) // iの初期化を追加
  {
    double dist = similarity_function(query.features, data[i]->features);
    distances.push_back(std::make_pair(data[i]->id, dist));
  }
  std::sort(distances.begin(), distances.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b)
            { return a.second < b.second; });
  std::vector<int> result;
  for (int i = 0; i < std::min(top_k, static_cast<int>(distances.size())); ++i)
  {
    result.push_back(distances[i].first);
  }
  return result;
}