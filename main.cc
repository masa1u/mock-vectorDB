#include "common/similarity_function.hh"
#include "common/dataset.hh"
#include "common/result.hh"
#include "index/ivf_flat.hh"
#include "index/flat.hh"
#include "index/ivf_fc_flat.hh"
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <atomic>
#include <thread>

int main()
{
  // YAMLファイルの読み込み
  YAML::Node config = YAML::LoadFile("../conf/sample.yaml");

  // 設定値の取得
  int ex_time = config["config"]["ex_time"].as<int>();
  int dimension = config["config"]["dimension"].as<int>();
  int num_vectors = config["config"]["num_vectors"].as<int>();
  int top_k = config["config"]["top_k"].as<int>();
  int num_threads = config["config"]["num_threads"].as<int>();
  int ivf_flat_nlist = config["config"]["ivf_flat"]["nlist"].as<int>();
  int ivf_flat_nprobe = config["config"]["ivf_flat"]["nprobe"].as<int>();
  int ivf_fc_nlist = config["config"]["ivf_fc_flat"]["nlist"].as<int>();
  int ivf_fc_nprobe = config["config"]["ivf_fc_flat"]["nprobe"].as<int>();
  double fuzzy_c_means_weight = config["config"]["ivf_fc_flat"]["fuzziness"].as<double>();

  // similarity_functionをユークリッド距離に設定
  functionSet();

  // ランダムベクトルの生成
  std::vector<Vector *> dataset = generateRandomVectors(dimension, num_vectors);

  bool start = false;
  bool quit = false;
  std::vector<int> readys(num_threads, 0);
  std::vector<std::thread> thv;

  IndexResults.resize(num_threads);

  // IVFFlatIndexの構築と検索
  IVFFlatIndex ivf_index(ivf_flat_nlist, dimension);
  ivf_index.buildIndex(dataset);

  for (size_t i = 0; i < num_threads; ++i)
  {
    thv.emplace_back(ivf_flat_worker, i, std::ref(readys[i]), std::ref(start), std::ref(quit), &ivf_index, dimension, top_k, ivf_flat_nprobe);
  }

  while (true)
  {
    bool failed = false;
    for (auto &re : readys)
    {
      if (!__atomic_load_n(&re, __ATOMIC_SEQ_CST))
      {
        failed = true;
        break;
      }
    }
    if (!failed)
    {
      break;
    }
  }

  __atomic_store_n(&start, true, __ATOMIC_SEQ_CST);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000 * ex_time));

  __atomic_store_n(&quit, true, __ATOMIC_SEQ_CST);

  for (auto &th : thv)
  {
    th.join();
  }

  // 結果の出力
  int throughput = 0;
  for (int i = 0; i < num_threads; i++)
  {
    throughput += IndexResults[i].queries_count;
  }
  std::cout << "Throughput: " << throughput / ex_time << " [qps]" << std::endl;
  std::cout << "recall: " << calculateRecall(dataset) << std::endl;

  // // FlatIndexの構築と検索
  // FlatIndex flat_index;
  // flat_index.buildIndex(dataset);
  // std::vector<int> flat_result = flat_index.search(query_vector->features, top_k);

  // // 結果の出力
  // for (int i = 0; i < top_k; i++)
  // {
  //   std::cout << "flat" << i + 1 << ": " << flat_result[i] << std::endl;
  // }

  // // FuzzyCMeansIndexの構築と検索
  // FuzzyCMeansIndex ivf_fc_index(ivf_fc_nlist, dimension, fuzzy_c_means_weight);
  // ivf_fc_index.buildIndex(dataset);
  // std::vector<int> ivf_fc_result = ivf_fc_index.search(query_vector->features, top_k, ivf_fc_nprobe);

  // // 結果の出力
  // for (int i = 0; i < top_k; i++)
  // {
  //   std::cout << "ivf_fc_flat" << i + 1 << ": " << ivf_fc_result[i] << std::endl;
  // }
}