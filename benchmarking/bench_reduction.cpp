#include "bench_reduction.h"

#include <iostream>
#include <random>

void
BenchmarkReduction::init(const boost::program_options::variables_map& vm)
{
  if (vm.count("chunksz"))
  {
    m_chunksz = vm["chunksz"].as<int>();
    std::cout << "Using chunksz " << m_chunksz << std::endl;
  }

  if (vm.count("newCode"))
  {
    m_useNewCode = true;
    std::cout << "Using new code" << std::endl;
  }
  
  size_t nUpperTimeLimit = 100;
  size_t nLowerTimeLimit = 2;
  
  size_t nPcfs = 10000;
  
  std::mt19937 gen;
  std::uniform_int_distribution nTimePointDist(nLowerTimeLimit, nUpperTimeLimit);
  std::exponential_distribution<float> timeDist;
  std::normal_distribution<float> valDist;
  
  m_pcfs32.resize(nPcfs);
  
  for (auto & pcf : m_pcfs32)
  {
    auto nTimePoints = nTimePointDist(gen);
    std::vector<mpcf::Point_f32> pts;
    pts.resize(nTimePoints);
    
    for (auto & pt : pts)
    {
      pt.t = timeDist(gen);
      pt.v = valDist(gen);
    }
    std::sort(pts.begin(), pts.end(), [](const mpcf::Point_f32& a, const mpcf::Point_f32& b){ return a.t < b.t; });
    pcf = mpcf::Pcf_f32(std::move(pts));
  }
  
}

void
BenchmarkReduction::run()
{
  mpcf::average(m_pcfs32, m_useNewCode, m_chunksz);
}
