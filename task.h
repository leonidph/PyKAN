#ifndef _TASK_
#define _TASK_

#include <map>
#include <thread>
#include <iostream>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <vector>
#include "json.hpp"
#include <torch/torch.h>

// Initialize with a permit count of 3
namespace nlohmann {

template<>
struct adl_serializer<torch::Tensor>
{
    static void to_json(json& j, const torch::Tensor& t)
    {
        auto cpu = t.cpu().contiguous();

        std::vector<float> data(
            cpu.data_ptr<float>(),
            cpu.data_ptr<float>() + cpu.numel());

        std::vector<int64_t> shape(
            cpu.sizes().begin(),
            cpu.sizes().end());

        j = {
            {"shape", shape},
            {"data", data}
        };
    }

    static void from_json(const json& j, torch::Tensor& t)
    {
        auto shape = j.at("shape").get<std::vector<int64_t>>();
        auto data  = j.at("data").get<std::vector<float>>();

        t = torch::from_blob(
                data.data(),
                shape,
                torch::kFloat32)
                .clone();
    }
};

}



class Semaphore {
public:
    explicit Semaphore(int count = 0)
        : count_(count) {}

    void release() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_one();
    }

    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return count_ > 0; });
        --count_;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
};

class TaskExecutor
{
public:
    typedef nlohmann::json (*tFunc)(nlohmann::json);

public:
    TaskExecutor(uint32_t maxThreads);
    ~TaskExecutor();

    void AddFunctionalty (const char* fname,tFunc fn)
    {
        functs_[fname] =fn;
    }

    uint64_t PushTask(const char* taskName,nlohmann::json& params);

    bool getTaskResult(uint64_t id,nlohmann::json& res);

    size_t TasksInQueue(){return prmQueue_.size();}
    bool hasJobs();

private:
    void Executor();
private:
    std::vector<std::thread*> threads_;
    std::map<std::string ,tFunc> functs_;
    uint64_t  taskID_;

    std::mutex mtxQueue_;
    std::queue<nlohmann::json> prmQueue_;
    std::set<uint64_t> taskInExecution_;
    std::atomic<bool> stopFlag_;
    std::map<uint64_t ,nlohmann::json> results_;

};

#endif
