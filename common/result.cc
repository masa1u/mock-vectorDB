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

void processThreRes(Result &thre_res, FlatIndex &bruteforce, int &local_retrieved, int &local_relevant)
{
  std::vector<int> flat_result;
  for (const auto &res : thre_res.search_results)
  {
    flat_result = bruteforce.search(res.first.features, res.second.size());
    local_retrieved += res.second.size();
    for (const auto element : res.second)
    {
      if (std::find(flat_result.begin(), flat_result.end(), element) != flat_result.end())
      {
        local_relevant++;
      }
    }
  }
}

std::pair<int, int> calculateRelevantAndRetrieved(std::vector<Vector *> dataset)
{
  int total_relevant = 0;
  int total_retrieved = 0;

  FlatIndex bruteforce;
  bruteforce.buildIndex(dataset);

  std::vector<std::thread> threads;
  std::vector<int> local_retrieved(IndexResults.size(), 0);
  std::vector<int> local_relevant(IndexResults.size(), 0);

  for (size_t i = 0; i < IndexResults.size(); ++i)
  {
    threads.emplace_back(processThreRes, std::ref(IndexResults[i]), std::ref(bruteforce), std::ref(local_retrieved[i]), std::ref(local_relevant[i]));
  }

  for (auto &thread : threads)
  {
    thread.join();
  }

  for (size_t i = 0; i < IndexResults.size(); ++i)
  {
    total_retrieved += local_retrieved[i];
    total_relevant += local_relevant[i];
  }

  return std::make_pair(total_relevant, total_retrieved);
}