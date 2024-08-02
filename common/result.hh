#pragma once

#include "dataset.hh"
#include "../index/flat.hh"
#include <vector>
#include <utility>
#include <thread>

class Result
{
public:
  int queries_count;
  std::vector<std::pair<Vector, std::vector<int>>> search_results; // クエリと検索結果kのペア
};

extern std::vector<Result> IndexResults;

double calculateRecall(std::vector<Vector *> dataset);
std::pair<int, int> calculateRelevantAndRetrieved(std::vector<Vector *> dataset);