#include <iostream>
#include <vector>
#include <cmath>

#include "similarity_function.hh"

double (*similarity_function)(const std::vector<double> &, const std::vector<double> &);

void functionSet()
{
  similarity_function = euclideanDistance;
}

bool vectorSize(const std::vector<double> &v1, const std::vector<double> &v2)
{
  if (v1.size() != v2.size())
  {
    std::cerr << "Error: Vectors must be of the same size." << std::endl;
    return false;
  }
  else
  {
    return true;
  }
}

double norm(const std::vector<double> &v)
{
  double sum = 0;
  for (double x : v)
  {
    sum += x * x;
  }
  return std::sqrt(sum);
}

double dotProduct(const std::vector<double> &v1, const std::vector<double> &v2)
{
  if (!vectorSize(v1, v2))
    return 0;

  double result = 0;
  for (size_t i = 0; i < v1.size(); ++i)
  {
    result += v1[i] * v2[i];
  }
  return result;
}

double cosineSimilarity(const std::vector<double> &v1, const std::vector<double> &v2)
{
  if (!vectorSize(v1, v2))
    return 0;

  double dot = 0;
  for (size_t i = 0; i < v1.size(); ++i)
  {
    dot += v1[i] * v2[i];
  }

  double norm1 = norm(v1);
  double norm2 = norm(v2);
  double denominator = norm1 * norm2;

  if (denominator == 0)
  {
    std::cerr << "Error: One or both vectors are zero vectors." << std::endl;
    return 0; // ノルムが0の場合はエラーとして0を返す
  }

  return dot / denominator;
}

double hammingDistance(const std::vector<double> &v1, const std::vector<double> &v2)
{
  if (!vectorSize(v1, v2))
    return 0;

  double distance = 0;
  for (size_t i = 0; i < v1.size(); ++i)
  {
    if (v1[i] != v2[i])
    {
      ++distance;
    }
  }
  return distance;
}

double euclideanDistance(const std::vector<double> &v1, const std::vector<double> &v2)
{
  if (!vectorSize(v1, v2))
    return 0;

  double sum = 0;
  for (size_t i = 0; i < v1.size(); ++i)
  {
    sum += std::pow(v1[i] - v2[i], 2);
  }

  return std::sqrt(sum);
}