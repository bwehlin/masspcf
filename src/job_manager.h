#ifndef MPCF_JOB_MANAGER_H
#define MPCF_JOB_MANAGER_H

#include <taskflow/taskflow.hpp>
#include <unordered_map>
#include <future>
#include <mutex>

namespace mpcf
{
  class Job
  {
  public:
    Job()
      : m_status(std::future_status::deferred)
    { }

  private:
    std::future_status m_status;
  };

  class JobManager
  {
  public:
    using job_id = long;

    std::future_status get_status(job_id id) const
    {
      std::lock_guard<std::mutex> lock(m_mutex);

    }

    Job take(job_id id)
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_jobs.find(id);
      Job job = std::move(it->second);
      m_jobs.erase(it);
      return job;
    }

  private:
    tf::Executor m_executor;
    std::unordered_map<job_id, std::future<>> m_jobs;
    mutable std::mutex m_mutex;
  };
}

#endif
