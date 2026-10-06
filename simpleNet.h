#pragma once
#include <torch/torch.h>
#include <torch/script.h> // Core LibTorch header for TorchScript

class SimpleNet : public torch::nn::Module
{
public:
    SimpleNet();
    void Load();
    void Save();
    torch::Tensor forward(torch::Tensor x);
private:

    torch::nn::Conv2d conv1, conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

