#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <chrono>

#include <boost/program_options.hpp>

#include "bench_reduction.h"

namespace
{
  std::unordered_map<std::string, std::unique_ptr<Benchmark>> tests;
  
  void show_help(const boost::program_options::options_description& desc)
  {
    std::cout << desc << "\n\nAvailable tests:\n\n";
    for (auto const & test : tests)
    {
      std::cout << "  " << test.first << '\n';
    }
    std::cout << std::flush;
  }
  
  void create_tests()
  {
    tests["reduction"] = std::make_unique<BenchmarkReduction>();
  }
}

int main(int argc, char** argv)
{
  namespace po = boost::program_options;
  
  try
  {
    create_tests();
    
    po::options_description desc("Allowed options");
    
    std::string testName;
    int nReps;
    
    desc.add_options()
      ("help,h", "shows this message")
      ("reps,r", po::value<int>(&nReps)->default_value(1), "number of repetitions")
      ("test", po::value<std::string>(&testName), "specifies which test to run")
      ("chunksz", po::value<int>(), "chunk size")
      ("newCode", "use new code")
    ;
    
    
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
      show_help(desc);
      return EXIT_FAILURE;
    }
    
    auto it = tests.find(testName);
    if (it == tests.end())
    {
      show_help(desc);
      return EXIT_FAILURE;
    }
    
    auto & test = *it->second;
    
    if (nReps < 1ul)
    {
      std::cerr << "Warning! nReps must be >= 1. Setting nReps to 1\n";
      nReps = 1;
    }
    
    test.init(vm);
    using clock_t = std::chrono::steady_clock;
    
    using fms = std::chrono::duration<float, std::chrono::milliseconds::period>;
    std::vector<fms> times;
    times.resize(nReps);
    for (auto i = 0; i < nReps; ++i)
    {
      auto start = clock_t::now();
      test.run();
      auto end = clock_t::now();
      times[i] = end - start;
    }
    
    auto timeTaken = std::accumulate(times.begin(), times.end(), fms(0));
    auto nf = static_cast<float>(times.size());
    
    auto mean = timeTaken.count() / nf;
    auto sd = std::accumulate(times.begin(), times.end(), 0.f, [mean, nf](float acc, fms t){
        return acc + ((t.count() - mean) * (t.count() - mean) / nf);
    });
    std::cout << "Finished " << nReps << " rep(s) in " << timeTaken.count() << "ms" << " avg: " << mean << " +- " << sd << " ms" << std::endl;
    
    return EXIT_SUCCESS;
  }
  catch (const std::exception& ex)
  {
    std::cerr << "Caught exception: " << ex.what() << '\n';
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception!\n";
  }
  
  return EXIT_FAILURE;
}