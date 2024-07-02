#include "dataset.hh"

Vector *createRandomVector(int dimension)
{
  // 乱数生成器の準備
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-10.0, 10.0); // -10.0から10.0の範囲の乱数を生成する

  // ベクトルを生成
  std::vector<double> vec;
  vec.reserve(dimension);

  // ランダムな特徴量を生成してベクトルを作成
  for (int j = 0; j < dimension; ++j)
  {
    vec.push_back(dis(gen));
  }

  return new Vector(vec);
}

std::vector<Vector *> generateRandomVectors(int dimension, int num_vectors)
{
  std::vector<Vector *> dataset;
  dataset.reserve(num_vectors);

  for (int i = 0; i < num_vectors; ++i)
  {
    Vector *vec = createRandomVector(dimension);
    dataset.push_back(vec);
  }

  return dataset;
}