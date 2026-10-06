#include "chebkan.hpp"



ChebyshevLayerImpl::ChebyshevLayerImpl(int64_t in_f,int64_t out_f,int64_t deg):
    in_features_(in_f),out_features_(out_f),degree_(deg)
{
    weights_=register_parameter("weights",torch::empty({out_f,in_f,deg+1}));
    bias_=register_parameter("bias",torch::zeros({out_f}));
    torch::nn::init::xavier_uniform_(weights_);
}

torch::Tensor ChebyshevLayerImpl::compute_basis(const torch::Tensor& x)
{
    auto xn=torch::clamp(x,-1.0,1.0);
    std::vector<torch::Tensor> b;
    b.push_back(torch::ones_like(xn));
    if(degree_>=1)
        b.push_back(xn);
    for(int64_t k=2;k<=degree_;++k)
        b.push_back(2.0*xn*b[k-1]-b[k-2]);

    return torch::stack(b,-1);}


torch::Tensor ChebyshevLayerImpl::forward(const torch::Tensor& x)
{
    auto basis=compute_basis(x);
    auto edge=torch::einsum("bid,oid->boi",{basis,weights_});
    auto y=edge.sum(-1);
    y+=bias_;
    return y;
}


ChebKANImpl::ChebKANImpl(int64_t inputSize,int64_t hiddenSize,int64_t outputSize,int64_t deg,int64_t allLayers)
{
    layers_=register_module("layers",torch::nn::ModuleList());
    layers_->push_back(ChebyshevLayer(inputSize,hiddenSize,deg));
    for(int64_t i=0;i<allLayers-2;++i)
    {
        layers_->push_back(ChebyshevLayer(hiddenSize,hiddenSize,deg));
    }
    layers_->push_back(ChebyshevLayer(hiddenSize,outputSize,deg));
}



torch::Tensor ChebKANImpl::forward(torch::Tensor x)
{
    size_t last=layers_->size()-1;
    for(size_t i=0;i<last;++i)
    {
        auto* l=layers_[i]->as<ChebyshevLayerImpl>();
        auto h=torch::silu(l->forward(x));
        if(h.size(1)==x.size(1))
            x=x+h;
        else
            x=h;
    }
    auto* out=layers_[last]->as<ChebyshevLayerImpl>();
    return out->forward(x);
}
