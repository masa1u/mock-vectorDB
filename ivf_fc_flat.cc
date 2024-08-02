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

int main(int argc, char *argv[])
{
  // YAMLファイルの読み込み
  std::string input_filename = argv[1];
  YAML::Node config = YAML::LoadFile(input_filename);

  // 設定値の取得
  int ex_time = config["config"]["ex_time"].as<int>();
  int num_runs = config["config"]["num_runs"].as<int>();
  int dimension = config["config"]["dimension"].as<int>();
  int num_vectors = config["config"]["num_vectors"].as<int>();
  int top_k = config["config"]["top_k"].as<int>();
  int num_threads = config["config"]["num_threads"].as<int>();
  int ivf_fc_nlist = config["config"]["ivf_fc_flat"]["nlist"].as<int>();
  int upper_nprobe = config["config"]["ivf_fc_flat"]["upper_nprobe"].as<int>();
  int lower_nprobe = config["config"]["ivf_fc_flat"]["lower_nprobe"].as<int>();
  int step_nprobe = config["config"]["ivf_fc_flat"]["step_nprobe"].as<int>();
  double fuzzy_c_means_weight = config["config"]["ivf_fc_flat"]["fuzziness"].as<double>();
  double upper_threshold = config["config"]["ivf_fc_flat"]["upper_threshold"].as<double>();
  double lower_threshold = config["config"]["ivf_fc_flat"]["lower_threshold"].as<double>();
  double step_threshold = config["config"]["ivf_fc_flat"]["step_threshold"].as<double>();

  // similarity_functionをユークリッド距離に設定
  functionSet();

  // ランダムベクトルの生成
  std::vector<Vector *> dataset = generateRandomVectors(dimension, num_vectors);
  FuzzyCMeansIndex ivf_fc_index(ivf_fc_nlist, dimension, fuzzy_c_means_weight);
  ivf_fc_index.clustering(dataset);
  for (double threshold = lower_threshold; threshold <= upper_threshold; threshold += step_threshold)
  {
    ivf_fc_index.buildIndex(dataset, threshold);
    for (int ivf_fc_nprobe = lower_nprobe; ivf_fc_nprobe <= upper_nprobe; ivf_fc_nprobe += step_nprobe)
    {
      for (int _ = 0; _ < num_runs; _++)
      {
        bool start = false;
        bool quit = false;
        std::vector<int> readys(num_threads, 0);
        std::vector<std::thread> ivf_fc_thv;
        IndexResults.resize(num_threads);

        for (size_t i = 0; i < num_threads; ++i)
        {
          ivf_fc_thv.emplace_back(ivf_fc_flat_worker, i, std::ref(readys[i]), std::ref(start), std::ref(quit), &ivf_fc_index, dimension, top_k, ivf_fc_nprobe);
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

        for (auto &th : ivf_fc_thv)
        {
          th.join();
        }

        int throughput = 0;
        for (int i = 0; i < num_threads; i++)
        {
          throughput += IndexResults[i].queries_count;
        }
        std::pair<int, int> result = calculateRelevantAndRetrieved(dataset);
        std::cout << "threshold: " << threshold << " nprobe: " << ivf_fc_nprobe << std::endl;
        std::cout << "[IVF_FC]Throughput: " << throughput / ex_time << " [qps]" << result.second / ex_time << std::endl;
        std::cout << "[IVF_FC]Recall: " << double(result.first) / result.second << std::endl;
        IndexResults.clear();
      }
    }
    ivf_fc_index.clearClusters();
  }
}