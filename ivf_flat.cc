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
#include <fstream>

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
  int ivf_flat_nlist = config["config"]["ivf_flat"]["nlist"].as<int>();
  int upper_nprobe = config["config"]["ivf_flat"]["upper_nprobe"].as<int>();
  int lower_nprobe = config["config"]["ivf_flat"]["lower_nprobe"].as<int>();
  int step_nprobe = config["config"]["ivf_flat"]["step_nprobe"].as<int>();

  // similarity_functionをユークリッド距離に設定
  functionSet();

  // ランダムベクトルの生成
  std::vector<Vector *> dataset = generateRandomVectors(dimension, num_vectors);

  // IVFFlatIndexの構築と検索
  IVFFlatIndex ivf_index(ivf_flat_nlist, dimension);
  ivf_index.buildIndex(dataset);
  for (int ivf_flat_nprobe = lower_nprobe; ivf_flat_nprobe <= upper_nprobe; ivf_flat_nprobe += step_nprobe)
  {
    for (int _ = 0; _ < num_runs; _++)
    {
      bool start = false;
      bool quit = false;
      std::vector<int> readys(num_threads, 0);
      std::vector<std::thread> ivf_flat_thv;

      IndexResults.resize(num_threads);

      for (size_t i = 0; i < num_threads; ++i)
      {
        ivf_flat_thv.emplace_back(ivf_flat_worker, i, std::ref(readys[i]), std::ref(start), std::ref(quit), &ivf_index, dimension, top_k, ivf_flat_nprobe);
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

      for (auto &th : ivf_flat_thv)
      {
        th.join();
      }

      // 結果の出力
      int throughput = 0;
      for (int i = 0; i < num_threads; i++)
      {
        throughput += IndexResults[i].queries_count;
      }
      std::pair<int, int> result = calculateRelevantAndRetrieved(dataset);
      std::cout << "[IVF_FLAT]Throughput: " << throughput / ex_time << " [qps]" << result.second / ex_time << std::endl;
      std::cout << "[IVF_FLAT]Recall: " << double(result.first) / result.second << std::endl;

      // CSVファイルへの書き込み
      std::ofstream csv_file(argv[2], std::ios::app);
      csv_file << ex_time << "," << _ << "," << dimension << "," << num_vectors << "," << top_k << "," << num_threads << "," << ivf_flat_nlist << "," << ivf_flat_nprobe << ",,,,," << throughput / ex_time << "," << result.first << "," << result.second << "\n";
      csv_file.close();

      IndexResults.clear();
    }
  }
}