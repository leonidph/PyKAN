#ifndef _KAN_
#define _KAN_

#include <torch/torch.h>

class KANLayerImpl : public torch::nn::Module {
public:
    int in_features_, out_features_, num_knots_;
    int spline_degree_ = 3;

    torch::Tensor base_weight_, spline_weight_, spline_scale_, knots_;
    torch::nn::SiLU silu_{nullptr};

    KANLayerImpl(int in_f,int out_f,int knots=10);

    void build_uniform_knots();
    void refine_grid(const torch::Tensor& samples);
    torch::Tensor bspline_basis(const torch::Tensor& x,int degree);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(KANLayer);

typedef torch::Tensor (*tSigmoid)(const torch::Tensor& );
class KANNetImpl : public torch::nn::Module
{
    tSigmoid sigmoid;

public:
    std::vector<KANLayer> layers_;

    KANNetImpl(size_t layers,size_t in,size_t out,size_t internalSize=16,size_t knots=10)
    {
        sigmoid =  torch::tanh;
        //sigmoid =torch::sigmoid;
        //sigmoid= torch::sqrt(torch::sigmoid(x))

        char name[20]={0};
        KANLayer layer0{nullptr}; layer0 = register_module("layer1",KANLayer(in,internalSize,knots));
        layers_.push_back(layer0);

        for(size_t l=2;l<layers;l++)
        {
            KANLayer layerH{nullptr};
            sprintf(name,"layer%ld",l);
            layerH = register_module(name,KANLayer(internalSize,internalSize,knots));
            layers_.push_back(layerH);
        }

        KANLayer layerO{nullptr};
        sprintf(name,"layer%ld",layers);
        layerO = register_module(name,KANLayer(internalSize,out,knots));
        layers_.push_back(layerO);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        for(size_t i=0;i<layers_.size()-1;i++)
        {
            x = layers_[i]->forward(x);
            x = sigmoid(x);
        }
        x = layers_[layers_.size()-1]->forward(x);
        return x;
    }
};

TORCH_MODULE(KANNet);

#endif// _KAN_
