#pragma once
#include <torch/torch.h>

class RegressionDataset:
        public torch::data::datasets::Dataset<RegressionDataset>
{
public:
    RegressionDataset(torch::Tensor x,torch::Tensor y):
        x_(std::move(x)),y_(std::move(y))
    {}

    torch::data::Example<> get(size_t i) override
    {
        return {x_[i],y_[i]};
    }
    torch::optional<size_t> size() const override
    {
        return x_.size(0);
    }
private:
    torch::Tensor x_,y_;
};

class ChebyshevLayerImpl: public torch::nn::Module
{
public:
    ChebyshevLayerImpl(int64_t,int64_t,int64_t);
    torch::Tensor forward(const torch::Tensor&);
private:
    torch::Tensor compute_basis(const torch::Tensor&);
    int64_t in_features_,out_features_,degree_;
    torch::Tensor weights_,bias_;
};
TORCH_MODULE(ChebyshevLayer);


class ChebKANImpl: public torch::nn::Module
{
public:
    ChebKANImpl(int64_t inputSize,int64_t hiddenSize,int64_t outputSize,int64_t deg,int64_t allLayers);
    torch::Tensor forward(torch::Tensor);
private:
    torch::nn::ModuleList layers_;
};

TORCH_MODULE(ChebKAN);
