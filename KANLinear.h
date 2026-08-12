#pragma once
#include <torch/torch.h>
#include <cmath>

class KANLinearImpl : public torch::nn::Module {
public:
    int in_features, out_features;
    int grid_size, spline_order;
    float scale_noise, scale_base, scale_spline;
    bool enable_standalone_scale_spline;
    float grid_eps;

    torch::Tensor grid;

    torch::Tensor base_weight;
    torch::Tensor spline_weight;
    torch::Tensor spline_scaler;

    torch::nn::SiLU base_activation;

    KANLinearImpl(
        int in_features,
        int out_features,
        int grid_size = 5,
        int spline_order = 3,
        float scale_noise = 0.1,
        float scale_base = 1.0,
        float scale_spline = 1.0,
        bool enable_standalone_scale_spline = true,
        float grid_eps = 0.02,
        std::pair<float,float> grid_range = {-1.0f, 1.0f}
    );

    void reset_parameters();
    torch::Tensor b_splines(const torch::Tensor& x);
    torch::Tensor curve2coeff(const torch::Tensor& grid, const torch::Tensor& noise);
    torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(KANLinear);

