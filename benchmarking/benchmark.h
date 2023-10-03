#ifndef BENCH_BENCHMARK_H
#define BENCH_BENCHMARK_H

#include <boost/program_options.hpp>

class Benchmark
{
public:
  virtual void init(const boost::program_options::variables_map& /* args */) { }
  virtual void run() = 0;
};

#endif
