#pragma once

#include <vector>

class Vector
{
public:
  std::vector<double> features;

  Vector(const std::vector<double> &features) : features(features) {}
};

Vector *
createRandomVector(int dimension);

std::vector<Vector *> generateRandomVectors(int dimension, int num_vectors);