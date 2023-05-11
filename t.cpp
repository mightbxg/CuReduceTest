#include <TestFuncs/TicToc.hpp>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "myreduce.h"

using namespace std;

int main(int argc, char** argv) {
  const size_t n = 10000;
  const size_t test_num = 2;
  vector<float> data;
  data.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    data.push_back(i);
  }
  auto res = accumulate(data.begin(), data.end(), 0.0);
  cout << fixed;
  cout << "result: " << res << endl;

  // -------------------------------------------
  cout << MyReduce(data.data(), n) << endl;
  float ret1 = 0.0f;
  for (size_t i = 0; i < test_num; ++i) {
    dbg::TicToc::ScopedTimer st("MyReduce");
    ret1 += MyReduce(data.data(), n);
  }

  // -------------------------------------------
  cout << ThReduce(data.data(), n) << endl;
  float ret2 = 0.0f;
  for (size_t i = 0; i < test_num; ++i) {
    dbg::TicToc::ScopedTimer st("ThReduce");
    ret2 += ThReduce(data.data(), n);
  }
}