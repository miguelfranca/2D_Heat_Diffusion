#pragma once
#include <array>

void prepare(float* h_O, bool* h_barriers, const int nx, const int ny);
void launchKernel(const int step, const int outputEvery, float* h_O, bool * h_barriers);
void addHeat(const int x, const int y, const float amount, const float radius);
void addBarrier(const int x, const int y, const float radius);
void finalize();

std::array<int, 3> scalarToRGB(double value);