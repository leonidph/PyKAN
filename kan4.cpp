#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "kan4.h"

class KAN4LayerImpl : public torch::nn::Module
{
public:
    int in_features_;
    int out_features_;
    int num_knots_;
    int spline_degree_ = 3;

    torch::Tensor base_weight_;
    torch::Tensor spline_weight_;
    torch::Tensor spline_scale_;
    torch::Tensor knots_;

    torch::nn::SiLU silu_{nullptr};

    KAN4LayerImpl(
            int in_features,
            int out_features,
            int num_knots = 10)
        :
          in_features_(in_features),
          out_features_(out_features),
          num_knots_(num_knots),
          silu_(torch::nn::SiLU())
    {
        register_module("silu", silu_);

        base_weight_ =
                register_parameter(
                    "base_weight",
                    torch::randn(
                        {out_features_, in_features_}) * 0.1);

        build_uniform_knots();

        int num_basis =
                num_knots_ + spline_degree_ - 1;

        spline_weight_ =
                register_parameter(
                    "spline_weight",
                    torch::randn(
                        {
                            out_features_,
                            in_features_,
                            num_basis
                        }) * 0.01);

        spline_scale_ =
                register_parameter(
                    "spline_scale",
                    torch::ones(
                        {
                            out_features_,
                            in_features_
                        }));
    }

    void build_uniform_knots()
    {
        std::vector<float> k;

        for(int i=0;i<spline_degree_;++i)
            k.push_back(-2.0f);

        for(int i=0;i<num_knots_;++i)
        {
            k.push_back(
                        -2.0f +
                        4.0f * i /
                        (num_knots_ - 1));
        }

        for(int i=0;i<spline_degree_;++i)
            k.push_back(2.0f);

        knots_ =
                torch::tensor(
                    k,
                    torch::kFloat32);
    }

    void refine_grid(
            const torch::Tensor& samples)
    {
        auto sorted =
                std::get<0>(
                    samples.flatten()
                    .cpu()
                    .sort());

        float xmin =
                sorted[0].item<float>();

        float xmax =
                sorted[sorted.size(0)-1]
                .item<float>();

        std::vector<float> k;

        for(int i=0;i<spline_degree_;++i)
            k.push_back(xmin);

        for(int i=0;i<num_knots_;++i)
        {
            int idx =
                    (sorted.size(0)-1)
                    * i /
                    (num_knots_-1);

            k.push_back(
                        sorted[idx]
                        .item<float>());
        }

        for(int i=0;i<spline_degree_;++i)
            k.push_back(xmax);

        knots_ =
                torch::tensor(
                    k,
                    torch::kFloat32);
    }

    torch::Tensor bspline_basis(
            const torch::Tensor& x,
            int degree)
    {
        auto knots =
                knots_.to(x.device());

        auto xe =
                x.unsqueeze(-1);

        int M =
                knots.size(0);

        int N0 =
                M - 1;

        std::vector<torch::Tensor> basis;

        for(int i=0;i<N0;++i)
        {
            basis.push_back(
                        ((xe >= knots[i]) &
                         (xe < knots[i+1]))
                    .to(torch::kFloat32));
        }

        for(int d=1; d<=degree; ++d)
        {
            std::vector<torch::Tensor> next;

            for(int i=0;i<N0-d;++i)
            {
                auto ld =
                        knots[i+d] -
                        knots[i];

                auto rd =
                        knots[i+d+1] -
                        knots[i+1];

                auto left =
                        torch::zeros_like(
                            basis[i]);

                auto right =
                        torch::zeros_like(
                            basis[i]);

                if(ld.item<float>() > 1e-8f)
                {
                    left =
                            ((xe-knots[i]) /
                             ld) * basis[i];
                }

                if(rd.item<float>() > 1e-8f)
                {
                    right =
                            ((knots[i+d+1]-xe)
                            / rd) *
                            basis[i+1];
                }

                next.push_back(
                            left + right);
            }

            basis = std::move(next);
        }

        return torch::cat(
                    basis,
                    -1);
    }

    torch::Tensor forward(
            torch::Tensor x)
    {
        auto base_part =
                torch::matmul(
                    silu_->forward(x),
                    base_weight_.t());

        auto basis =
                bspline_basis(
                    x,
                    spline_degree_);

        auto scaled_weight =
                spline_weight_ *
                spline_scale_.unsqueeze(-1);

        auto spline_part =
                torch::einsum(
                    "bin,oin->bo",
                    {
                        basis,
                        scaled_weight
                    });

        return base_part +
                spline_part;
    }
};

TORCH_MODULE(KAN4Layer);

class KAN4NetImpl :
        public torch::nn::Module
{
    typedef torch::Tensor (*tNorm)(const torch::Tensor&);
    tNorm pNormal=nullptr;
public:

    KAN4Layer layer1{nullptr};
    KAN4Layer layer2{nullptr};
    KAN4Layer layer3{nullptr};

    void setNormalization(std::string func)
    {
        if(func=="th")
        {
            pNormal = torch::tanh;
        }
        else    if(func=="ss")
        {
            pNormal = torch::tanh;
        }
        else
        {assert(false);}
    }

    KAN4NetImpl()
    {
        pNormal = torch::tanh;
        //torch::nn::functional::softsign
        layer1 =register_module("layer1",KAN4Layer(4,64,10));
        layer2 =register_module("layer2",KAN4Layer(64,64,10));
        layer3 =register_module("layer3",KAN4Layer(64,1,10));
    }


    torch::Tensor forward(torch::Tensor x)
    {
        x =pNormal(layer1->forward(x));
        x =pNormal(layer2->forward(x));
        x = layer3->forward(x);
        return x;
    }
};

TORCH_MODULE(KAN4Net);

int main4(torch::Tensor X,torch::Tensor Y,std::string sNorm,int maxEpoch,float expLoss)
{

    //-------------------------------------------------
    // Model
    //-------------------------------------------------

    auto model =KAN4Net();
    model->setNormalization(sNorm);
    torch::optim::Adam optimizer(
                model->parameters(),
                torch::optim::AdamOptions(
                    1e-3));

    //-------------------------------------------------
    // Training
    //-------------------------------------------------

    for(int epoch=0;epoch<maxEpoch;++epoch)
    {
        optimizer.zero_grad();

        auto pred = model->forward(X);

        auto loss = torch::mse_loss(pred,Y);

        loss.backward();
        optimizer.step();
        if(epoch > 0 &&  epoch % 500 == 0)
        {
            model->layer1->refine_grid(X);
        }

        /*  if(epoch % 100 == 0)
        {
            std::cout
                    << "Epoch "
                    << epoch
                    << " Loss = "
                    << loss.item<float>()
                    << std::endl;
        }*/
        if(loss.item<float>() <expLoss)
        {break;}
    }

    //-------------------------------------------------
    // Test
    //-------------------------------------------------

    auto test = torch::tensor({
                                  {1.0f, 0.5f, 0.1f, 0.2f},
                                  {-1.0f, 1.5f, -0.5f, 2.0f},
                                  {0.0f, 0.0f, 0.0f, 0.0f}
                              });

    auto result = model->forward(test);

    std::cout
            << "\nPredictions:\n"
            << result
            << std::endl;

    torch::save(model,"kan_4d.pt");

    return 0;
}

nlohmann::json funcKAN4(nlohmann::json params)
{
    torch::Tensor X= params["X"];
    torch::Tensor Y= params["Y"];
    std::string sNorm =  params["norm"].get<std::string>();
    float loss = main4(X,Y,
                       sNorm ,
                       params["maxEpoch"].get<int>(),1e-5);

    std::string explanation= (std::string)"Normalization - " + sNorm +
                            " inp size - " + std::to_string(params["size"].get<int>());

    params["explanation"] =explanation;
    params["loss_result"] =loss;
    return params;
}


void CreateDataSet (int cnt, torch::Tensor& X, torch::Tensor& Y)
{
    auto x1 = torch::rand({cnt,1}) * 4 - 2;

    auto x2 = torch::rand({cnt,1}) * 4 - 2;
    auto x3 = torch::rand({cnt,1}) * 4 - 2;
    auto x4 = torch::rand({cnt,1}) * 4 - 2;

    X = torch::cat(
                {
                    x1,x2,x3,x4
                },1);
    Y =  torch::sin(3*x1)
            + torch::cos(2*x2)
            + 0.3*torch::sin(x3*x4)
            + 0.2*torch::pow(x1,2)
            - 0.1*torch::pow(x2,2)
            + 0.05*x3*x4 ;
}



void mainKan4(TaskExecutor &executor)
{
    torch::manual_seed(42);

    torch::Tensor X, Y;

    //-------------------------------------------------
    // Generate 4D training data
    //-------------------------------------------------



    nlohmann::json prm;
    std::set<uint64_t> tasksID;

/*

    CreateDataSet(1000,X,Y);
    {
        prm["size"]=1000;
        prm["X"]=X;
        prm["Y"]=Y;
        prm["maxEpoch"]=1000;
        prm["norm"]="th";
        tasksID.insert(executor.PushTask("KAN4",prm));
    }
    {
        prm["size"]=1000;
        prm["X"]=X;
        prm["Y"]=Y;
        prm["maxEpoch"]=1000;
        prm["norm"]="ss";
        tasksID.insert(executor.PushTask("KAN4",prm));
    }

    CreateDataSet(10000,X,Y);
    {
        prm["size"]=10000;
        prm["X"]=X;
        prm["Y"]=Y;
        prm["maxEpoch"]=1000;
        prm["norm"]="th";
        tasksID.insert(executor.PushTask("KAN4",prm));
    }
    {
        prm["size"]=10000;
        prm["X"]=X;
        prm["Y"]=Y;
        prm["maxEpoch"]=1000;
        prm["norm"]="ss";
        tasksID.insert(executor.PushTask("KAN4",prm));
    }
    CreateDataSet(100000,X,Y);
    {
        prm["size"]=100000;
        prm["X"]=X;
        prm["Y"]=Y;
        prm["maxEpoch"]=1000;
        prm["norm"]="th";
        tasksID.insert(executor.PushTask("KAN4",prm));
    }
  */  {
        CreateDataSet(100000,X,Y);

        prm["size"]=100000;
        prm["X"]=X;
        prm["Y"]=Y;
        prm["maxEpoch"]=1000;
        prm["norm"]="ss";
        tasksID.insert(executor.PushTask("KAN4",prm));
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


