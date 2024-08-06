#!/bin/bash

# 現在時刻の名前のCSVファイルを作成
timestamp=$(date +"%Y%m%d%H%M%S")
csv_file="result/${timestamp}.csv"
echo "ex_time, num_runs, dimension, num_vectors, top_k, num_threads, ivf_flat_nlist, ivf_flat_nprobe, ivf_fc_flat_nlist, fuzziness, ivf_fc_flat_nprobe, threshold, throughput, relevant, retrieved" > "$csv_file"

# confディレクトリ内の全てのyamlファイルに対してループ
for yaml_file in conf/*.yaml; do
  # 各yamlファイルに対して1回実行
  for i in {1..1}; do
    # buildディレクトリ内のmain実行ファイルを実行し、yamlファイルのパスを引数として渡す
    ./build/ivf_flat "$yaml_file" "$csv_file"
    ./build/ivf_fc_flat "$yaml_file" "$csv_file"
  done
done