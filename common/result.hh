#pragma once

#include "dataset.hh"
#include <vector>

class Result
{
public:
  int queries_count;
  std::vector<std::pair<Vector, std::vector<int>>> search_results; // クエリと検索結果kのペア
};

extern std::vector<Result> IndexResults;