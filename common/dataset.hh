#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <ctime>

class Vector
{
public:
  Vector() = default; // デフォルトコンストラクタ
  int id;
  std::vector<double> features;

  Vector(const std::vector<double> &features) : features(features) {}
};

Vector *
createRandomVector(int dimension);

std::vector<Vector *> generateRandomVectors(int dimension, int num_vectors);