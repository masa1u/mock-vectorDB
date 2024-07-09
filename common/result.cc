#include "result.hh"

std::vector<Result> IndexResults;

double calculateRecall(std::vector<Vector *> dataset)
{
  int total_relevant = 0;
  int total_retrieved = 0;

  FlatIndex bruteforce;
  bruteforce.buildIndex(dataset);

  std::vector<int> flat_result;
  for (const auto &thre_res : IndexResults)
  {
    for (const auto &res : thre_res.search_results)
    {
      flat_result = bruteforce.search(res.first.features, res.second.size());
      total_retrieved += res.second.size();
      for (const auto element : res.second)
      {
        if (std::find(flat_result.begin(), flat_result.end(), element) != flat_result.end())
        {
          total_relevant++;
        }
      }
    }
  }

  return (double)total_relevant / total_retrieved;
}

std::pair<int, int> calculateRelevantAndRetrieved(std::vector<Vector *> dataset)
{
  int total_relevant = 0;
  int total_retrieved = 0;

  FlatIndex bruteforce;
  bruteforce.buildIndex(dataset);

  std::vector<int> flat_result;
  for (const auto &thre_res : IndexResults)
  {
    for (const auto &res : thre_res.search_results)
    {
      flat_result = bruteforce.search(res.first.features, res.second.size());
      total_retrieved += res.second.size();
      for (const auto element : res.second)
      {
        if (std::find(flat_result.begin(), flat_result.end(), element) != flat_result.end())
        {
          total_relevant++;
        }
      }
    }
  }

  return std::make_pair(total_relevant, total_retrieved);
}