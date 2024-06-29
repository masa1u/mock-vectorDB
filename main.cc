#include "common/similarity_function.hh"
#include "common/dataset.hh"

#include <iostream>
#include <vector>

int main()
{
  std::vector<double> v1 = {2.2, 5.1, 1.9};
  std::vector<double> v2 = {-4.1, 1.4, -3.5};

  std::cout << "dot prodocut: " << dotProduct(v1, v2) << std::endl;
  std::cout << "cosine similarity: " << cosineSimilarity(v1, v2) << std::endl;
  std::cout << "hamming distance: " << hammingDistance(v1, v2) << std::endl;
}