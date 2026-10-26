
// KAN with Cubic B-Splines + SiLU branch + Adaptive Grid Refinement
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "task.h"
#include "kan.h"
#include "rnn.h"
#include "simpleFunc.h"
#include "kan4.h"

#include "chebkan.hpp"


float Run(size_t layers,size_t in,size_t out,size_t internalSize=16,size_t knots=10)
{
    torch::Tensor loss ;
    // Best result on 3,32,15
    auto X = torch::linspace(-2,2,5000).view({-1,1});
    auto Y = torch::sin(3*X) + 0.3*torch::pow(X,2) + 0.1*torch::tan(X);

    auto model = KANNet(layers,in,out,internalSize,knots);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-3));

    for(int epoch=0; epoch<10000; ++epoch)
    {
        opt.zero_grad();
        auto pred = model->forward(X);
        loss = torch::mse_loss(pred,Y);
        if(epoch%500 == 0)
        {
            //std::cout << "Epoch "<<epoch <<"\t Loss - "<<loss.item<float>()<<std::endl;
        }
        loss.backward();
        opt.step();
    }

    auto test = torch::tensor({{-1.5f},{-1.0f},{0.0f},{1.0f},{1.5f},{1.5f},{1.5f}});
    auto testR = torch::sin(3*test) + 0.3*torch::pow(test,2) + 0.1*torch::tan(test);
    //  std::cout << "Correct results :"<<std::endl<<testR<<std::endl<<std::endl;

    //    std::cout << "net results:"<<std::endl<< model->forward(test) << std::endl;
    //    torch::save(model,"kan_adaptive.pt");
    return loss.item<float>();
}


float Run2(torch::Tensor& X,torch::Tensor& Y,size_t layers,size_t in,size_t out,size_t internalSize=16,size_t knots=10,size_t maxEpoch=2000)
{
    torch::Tensor loss ;

    auto model = KANNet(layers,in,out,internalSize,knots);
    torch::optim::Adam opt(model->parameters(), 1e-3);

    for(int epoch=0; epoch<maxEpoch; ++epoch)
    {
        opt.zero_grad();


        auto pred = model->forward(X);
        loss = torch::mse_loss(pred,Y);
        if(epoch%500 == 0)
        {
            //std::cout << "Epoch "<<epoch <<"\t Loss - "<<loss.item<float>()<<std::endl;
        }
        loss.backward();
        opt.step();
    }


    //    torch::save(model,"kan_adaptive.pt");
    return loss.item<float>();
}

nlohmann::json funcKAN(nlohmann::json params)
{
    torch::Tensor X= params["X"];
    torch::Tensor Y= params["Y"];
    float loss = Run2(X,Y,
                      params["layers"].get<int>(),params["in"].get<int>(),
            params["out"].get<int>(),
            params["intSize"].get<int>(),params["knots"].get<int>(),
            params["maxEpoch"].get<int>());

    std::string explanation=
            "layers-" + std::to_string(params["layers"].get<int>() )+
            "  InternalSize - " + std::to_string(params["intSize"].get<int>()) +
            " Knots - " + std::to_string(params["knots"].get<int>());

    params["explanation"] =explanation;
    params["loss_result"] =loss;
    return params;
}

nlohmann::json funcRNN(nlohmann::json params)
{
    uint32_t realEpoch=0;
    float loss = runRNN(params["intSize"].get<int>(),params["layers"].get<int>(),params["maxEpoch"].get<int>(),realEpoch);
    std::string explanation=
            "layers-" + std::to_string(params["layers"].get<int>() )+
            " InternalSize - " + std::to_string(params["intSize"].get<int>())  +
            " Real Epochs - " +std::to_string(realEpoch) ;

    params["explanation"] =explanation;
    params["loss_result"] =loss;

    return params;
}


void mainKAN(TaskExecutor &executor)
{
    nlohmann::json prm;
    std::set<uint64_t> tasksID;

    // KAN data set
    auto X = torch::linspace(-2,3,1000).view({-1,1});
    auto Y = torch::sin(3*X);

    //      KAN
    std::vector<size_t> layers={3};
    std::vector<size_t> internalsizes={8};//{8,16,32,64,128,256};
    std::vector<size_t> knots={10};

    for(auto ls:layers)
        for(auto is:internalsizes)
            for(auto kn:knots)
            {
                prm["in"]=1;
                prm["out"]=1;
                prm["layers"]=ls;
                prm["intSize"] =is;
                prm["knots"] = kn;
                prm["maxEpoch"]=4000;
                nlohmann::json jX = X;
                prm["X"] =jX;
                prm["Y"] = Y;
                tasksID.insert(executor.PushTask("KAN",prm));
            }




    std::this_thread::sleep_for(std::chrono::seconds(2));
    while(executor.hasJobs())
    {
        for(uint64_t id:tasksID)
        {
            nlohmann::json res;
            if(executor.getTaskResult(id,res))
            {
                std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
                tasksID.erase(id);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << " ************************************************ "<<std::endl;
    for(uint64_t id:tasksID)
    {
        nlohmann::json res;
        executor.getTaskResult(id,res);
        std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
    }

}
\



nlohmann::json funcChebyshev(nlohmann::json params)
{
    torch::Device device(torch::cuda::is_available()?torch::kCUDA:torch::kCPU);

    torch::Tensor train_x  = params["train_x"];
    torch::Tensor train_y  = params["train_y"];

    auto dataset=RegressionDataset(train_x,train_y).map(torch::data::transforms::Stack<>());
    auto loader=torch::data::make_data_loader(std::move(dataset),torch::data::DataLoaderOptions().batch_size(64));
    float expLoss = 1e-5;

    ChebKAN model(params["input_size"].get<int>(),params["intSize"].get<int>(),1,10,params["layers"].get<int>());
    model->to(device);
    torch::optim::AdamW opt(model->parameters(),torch::optim::AdamWOptions(1e-3));
    float lastLoss;
    for(size_t e=0;e<params["maxEpoch"].get<int>();++e)
    {
        for(auto& batch:*loader)
        {
            auto bx=batch.data.to(device);
            auto by=batch.target.to(device);
            opt.zero_grad();
            auto loss=torch::mse_loss(model->forward(bx),by);
            lastLoss = loss.item<float>();
            loss.backward(); opt.step();
        }
        if(lastLoss<expLoss)
        {
            break;
        }
    }
    model->printWB();
    //std::cout<<"Done after "<<e<<" epochs, with loss - "<<lastLoss<<std::endl;
    std::string explanation=
            "layers-" + std::to_string(params["layers"].get<int>() )+
            "  InternalSize - " + std::to_string(params["intSize"].get<int>());


    params["explanation"] =explanation;
    params["loss_result"] =lastLoss;
    return params;

}

void mainChebyshevKAN(TaskExecutor &executor,int inputSize)
{
    std::set<uint64_t> tasksID;

    torch::manual_seed(42);
    torch::Tensor train_x ;
    torch::Tensor train_y ;
    torch::Tensor val_x;
    torch::Tensor val_y;

    torch::Device device(torch::cuda::is_available()?torch::kCUDA:torch::kCPU);
    if(1 == inputSize)
    {
        auto x = torch::linspace(-2,2,1000).view({-1,1});
        auto y = torch::sin(3*x);

        train_x   =x.narrow(0,0,1000);
        train_y   =y.narrow(0,0,1000);

        std::cout << x.sizes()<<std::endl;
        val_x     =x.narrow(0,800,200);
        val_y     =y.narrow(0,800,200);
    }
    else    if(2 == inputSize)
    {
        auto x = torch::linspace(-2,2,8000).view({-1,2});

        auto col1 = x.select(1, 0).unsqueeze(1);
        auto col2 = x.select(1, 1).unsqueeze(1);


        auto Y1 = torch::sin(3*col1) + 0.3*torch::pow(col1,2);
        auto Y2 = torch::sin(3*col2) + 0.3*torch::pow(col2,2);

        auto y = (Y1 + Y2 ) / 2.0;

        train_x   =x.narrow(0,0,2000);
        train_y   =y.narrow(0,0,2000);
        val_x     =x.narrow(0,2000,2000);
        val_y     =y.narrow(0,2000,2000);
    }
    else if(4== inputSize)
    {
        auto x1 = torch::rand({10000,1}) * 4 - 2;
        auto x2 = torch::rand({10000,1}) * 4 - 2;
        auto x3 = torch::rand({10000,1}) * 4 - 2;
        auto x4 = torch::rand({10000,1}) * 4 - 2;

        auto X = torch::cat({x1,x2,x3,x4},1);

        auto Y =
                torch::sin(3*x1)  + torch::cos(2*x2) + 0.3*torch::sin(x3*x4)
                + 0.2*torch::pow(x1,2) - 0.1*torch::pow(x2,2) + 0.05*x3*x4;

    }
    nlohmann::json params;

    std::vector<size_t> layers={3};
    std::vector<size_t> internalsizes={20};
    for(auto ls:layers)
        for(auto is:internalsizes)
        {
            params["input_size"] = inputSize;
            params["train_x"] = train_x;
            params["train_y"] = train_y;
            params["intSize"] = is;
            params["layers"] =ls;
            params["maxEpoch"] =20000;
            tasksID.insert(executor.PushTask("Chebyshev",params));
        }

    std::cout << "tasks in queue - "<<executor.TasksInQueue() << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    while(executor.hasJobs())
    {
        for(uint64_t id:tasksID)
        {
            nlohmann::json res;
            if(executor.getTaskResult(id,res))
            {
                std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
                tasksID.erase(id);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << " ************************************************ "<<std::endl;
    for(uint64_t id:tasksID)
    {
        nlohmann::json res;
        executor.getTaskResult(id,res);
        std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
    }

}


void mainRNN(  TaskExecutor &executor)
{
    nlohmann::json prm;
    std::set<uint64_t> tasksID;




    std::vector<size_t> layers={3,4,5,6};
    std::vector<size_t> internalsizes={8,16,32,64};
    for(auto ls:layers)
        for(auto is:internalsizes)
        {
            prm["intSize"]=is;
            prm["layers"]=ls;
            prm["maxEpoch"]=3000;
            tasksID.insert(executor.PushTask("RNN",prm));
        }


    std::this_thread::sleep_for(std::chrono::seconds(2));
    while(executor.hasJobs())
    {
        for(uint64_t id:tasksID)
        {
            nlohmann::json res;
            if(executor.getTaskResult(id,res))
            {
                std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
                tasksID.erase(id);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << " ************************************************ "<<std::endl;
    for(uint64_t id:tasksID)
    {
        nlohmann::json res;
        executor.getTaskResult(id,res);
        std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
    }

}

void mainSimpleNet(TaskExecutor &executor)
{
    std::set<uint64_t> tasksID;

    nlohmann::json prm;
    prm["maxEpoch"]=100;
    tasksID.insert(executor.PushTask("Simple",prm));

    std::this_thread::sleep_for(std::chrono::seconds(2));
    while(executor.hasJobs())
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for(uint64_t id:tasksID)
    {
        nlohmann::json res;
        executor.getTaskResult(id,res);
        std::cout << "Task "<< id <<"  result - "<<res["loss_result"].get<float>()<< " expl-"<<res["explanation"]<<" Time-"<<res["time"].get<int>()<<" sec"<<std::endl;
    }
}


int main()
{
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;

    TaskExecutor executor(2);
    executor.AddFunctionalty("KAN",funcKAN);
    executor.AddFunctionalty("KAN4",funcKAN4);
    executor.AddFunctionalty("RNN",funcRNN);
    executor.AddFunctionalty("Chebyshev",funcChebyshev);
    executor.AddFunctionalty("Simple",SimpleFunc);
    //mainKan4(executor);
    //mainChebyshevKAN(executor,1);

    mainKAN(executor);
    //mainSimpleNet(executor);
    //mainRNN(executor);
    return 0;
}
