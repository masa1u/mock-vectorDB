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

# 実行ファイルのビルド
add_executable(main main.cc)

# サブディレクトリの追加
add_subdirectory(common)

# main と common ライブラリのリンク
target_link_libraries(main PUBLIC common)