#!/bin/bash

cd build
make
cd ..

# confディレクトリ内の全てのyamlファイルに対してループ
for yaml_file in conf/*.yaml; do
  # 各yamlファイルに対して5回実行
  for i in {1..1}; do
    # buildディレクトリ内のmain実行ファイルを実行し、yamlファイルのパスを引数として渡す
    ./build/ivf_flat "$yaml_file"
    ./build/ivf_fc_flat "$yaml_file"
  done
done
