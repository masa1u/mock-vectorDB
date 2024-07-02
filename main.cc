#include "common/similarity_function.hh"
#include "common/dataset.hh"
#include "index/ivf_flat.hh"

#include <iostream>
#include <vector>

int main()
{
  functionSet(); // 現状はsimilarity_functionを内積に設定

  int dimension = 100;   // ベクトルの次元数
  int num_vectors = 100; // ベクトルの数
  std::vector<Vector *> dataset = generateRandomVectors(dimension, num_vectors);

  Vector *query_vector = createRandomVector(dimension);

  IVFFlatIndex index(5, dimension);
  index.buildIndex(dataset);

  for (int i = 0; i < num_vectors; i++)
  {
    std::cout << "(" << i << ")" << "similarity: " << similarity_function(query_vector->features, dataset[i]->features) << std::endl;
    std::cout << "(" << i << ")" << "dot prodocut: " << dotProduct(query_vector->features, dataset[i]->features) << std::endl;
    std::cout << "(" << i << ")" << "cosine similarity: " << cosineSimilarity(query_vector->features, dataset[i]->features) << std::endl;
    std::cout << "(" << i << ")" << "hamming distance: " << hammingDistance(query_vector->features, dataset[i]->features) << std::endl;
  }
}