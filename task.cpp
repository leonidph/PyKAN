#include "task.h"


TaskExecutor::TaskExecutor(uint32_t maxThreads):taskID_(0)
{
    stopFlag_=false;
    for(int i=0;i<maxThreads;i++)
    {
        std::thread* thr = new std::thread(&TaskExecutor::Executor,this);
        threads_.push_back(thr);
    }
}

TaskExecutor::~TaskExecutor()
{
    stopFlag_=true;
    for(std::thread* th:threads_)
    {
        th->join();
    }
}

void TaskExecutor::Executor()
{
    while(!stopFlag_)
    {
        mtxQueue_.lock();
        if(prmQueue_.empty())
        {
            mtxQueue_.unlock();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        nlohmann::json task = prmQueue_.front();
        prmQueue_.pop();
        taskInExecution_.insert( task["taskId"].get<int>());
        mtxQueue_.unlock();
        std::string fname = task["fncname"].get<std::string>();
        auto it =   functs_.find(fname);
        if(it == functs_.end())
        {
            assert(false);
        }
        tFunc fn = it->second;
        try
        {            
            auto start = std::chrono::steady_clock::now();

            nlohmann::json res = (*fn)(task);

            auto end = std::chrono::steady_clock::now();
            auto dt_sec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
            res["time"] = dt_sec;

            uint64_t id = task["taskId"].get<uint64_t>();

            mtxQueue_.lock();

            results_[id] = res;
            mtxQueue_.unlock();        
        }
        catch (...)
        {
            std::cout << "failed" <<std::endl;
        }

        mtxQueue_.lock();
        taskInExecution_.erase(taskInExecution_.find(task["taskId"].get<int>()));
        mtxQueue_.unlock();

    }
}

bool TaskExecutor::getTaskResult(uint64_t id,nlohmann::json& res)
{
    mtxQueue_.lock();

    auto it = results_.find(id);
    if(it == results_.end())
    {
        mtxQueue_.unlock();
        return false;
    }
    res = it->second;
    results_.erase(it);
    mtxQueue_.unlock();
    return true;
}

bool TaskExecutor::hasJobs()
{
    bool br = !taskInExecution_.empty() ||
            !prmQueue_.empty();
    return br;
}

uint64_t TaskExecutor::PushTask(const char* taskName,nlohmann::json& params)
{
    uint64_t id;
    mtxQueue_.lock();
    id=++taskID_;

    params["fncname"] = taskName;
    params["taskId"] = id;

    prmQueue_.push(params);
    mtxQueue_.unlock();
    return id;;
}
