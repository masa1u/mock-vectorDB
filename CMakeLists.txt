# CMake 最低限のバージョン要求
cmake_minimum_required(VERSION 3.29)

# プロジェクト名と言語設定
project(VectorSearch CXX)

# バージョン番号
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)

# コンパイラオプションの設定
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# yaml-cppの設定
find_package(PkgConfig REQUIRED)
pkg_search_module(YAML_CPP REQUIRED yaml-cpp)

# pthreadの設定
find_package(Threads REQUIRED)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${YAML_CPP_INCLUDE_DIRS})
link_directories("/usr/local/Cellar/yaml-cpp/0.8.0/lib")

# 実行ファイルのビルド
add_executable(ivf_fc_flat ivf_fc_flat.cc)
add_executable(ivf_flat ivf_flat.cc)

# サブディレクトリの追加
add_subdirectory(common)
add_subdirectory(index)

# main と common ライブラリのリンク
target_link_libraries(ivf_fc_flat PUBLIC common)
target_link_libraries(ivf_fc_flat PUBLIC index)
target_link_libraries(ivf_flat PUBLIC common)
target_link_libraries(ivf_flat PUBLIC index)

# yaml-cppとpthreadライブラリのリンク
target_link_libraries(ivf_fc_flat PRIVATE yaml-cpp)
target_link_libraries(ivf_fc_flat PRIVATE Threads::Threads)
target_link_libraries(ivf_flat PRIVATE yaml-cpp)
target_link_libraries(ivf_flat PRIVATE Threads::Threads)
