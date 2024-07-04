#pragma once

#include <vector>
#include <algorithm>

#include "../common/dataset.hh"
#include "../common/similarity_function.hh"

class FlatIndex
{
public:
  FlatIndex() = default;
  void buildIndex(const std::vector<Vector *> dataset);
  std::vector<int> search(const Vector &query, int top_k);

private:
  std::vector<Vector *> data;
};