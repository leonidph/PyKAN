
// main.cpp
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>


#include "DataSrc.h"

// -----------------------------
// KANLayer: base linear + spline
// -----------------------------
struct KANLayerImpl : torch::nn::Module {
    int in_features;
    int out_features;
    int grid_size;
    double sigma;

    // base linear transform
    torch::nn::Linear base_linear{nullptr};

    // spline weights: [out_features, in_features, grid_size]
    torch::Tensor spline_weight;

    // grid centers: [grid_size]
    torch::Tensor grid_centers;

    KANLayerImpl(int in_f, int out_f, int grid, double sigma_)
        : in_features(in_f),
          out_features(out_f),
          grid_size(grid),
          sigma(sigma_),
          base_linear(register_module("base_linear",
                                      torch::nn::Linear(in_f, out_f))) {

        // initialize spline weights
        spline_weight = register_parameter(
                    "spline_weight",
                    torch::randn({out_features, in_features, grid_size}) * 0.01
                    );

        // grid centers in [-1, 1]
        grid_centers = torch::linspace(-1.0, 1.0, grid_size);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        // x: [batch, in_features]
        auto base_out = torch::silu(base_linear(x)); // base nonlinear part

        // normalize x to [-1, 1] (simple min/max-free heuristic)
        auto x_norm = torch::tanh(x); // [batch, in_features]

        // compute Gaussian basis for each input dim and grid point
        // result: [batch, in_features, grid_size]
        auto x_expanded = x_norm.unsqueeze(-1);              // [batch, in_features, 1]
        auto centers = grid_centers.view({1, 1, grid_size}); // [1, 1, grid_size]

        auto diff = x_expanded - centers;                    // [batch, in_features, grid_size]
        auto basis = torch::exp(- (diff * diff) / (2 * sigma * sigma));

        // spline_weight: [out_features, in_features, grid_size]
        // we want: spline_out[b, o] = sum_{i,g} basis[b,i,g] * spline_weight[o,i,g]
        auto w = spline_weight.unsqueeze(0); // [1, out_features, in_features, grid_size]
        auto b = basis.unsqueeze(1);        // [batch, 1, in_features, grid_size]

        auto prod = b * w;                  // [batch, out_features, in_features, grid_size]
        auto spline_out = prod.sum({2, 3}); // sum over in_features and grid_size -> [batch, out_features]

        // total output
        return base_out + spline_out;
    }
};
TORCH_MODULE(KANLayer);

// -----------------------------
// KANNet: simple 2-layer KAN
// -----------------------------
struct KANNetImpl : torch::nn::Module
{
    std::vector<KANLayer> layers_;
    uint32_t layersCnt_;

    KANNetImpl(uint32_t layersCnt,int input_dim, int hidden_dim, int output_dim,
               int grid_size, double sigma)
    {
        layersCnt_  =layersCnt;

        char name[20]={};
        sprintf(name,"layer%d",1);
        auto layer = register_module(name,KANLayer(input_dim, hidden_dim, grid_size, sigma));
        layers_.push_back(layer);

        for(size_t i=1;i<layersCnt_-1;i++)
        {
            sprintf(name,"layer%d",i+1);
            auto layer = register_module(name,KANLayer(hidden_dim, hidden_dim, grid_size, sigma));
            layers_.push_back(layer);
        }


        sprintf(name,"layer%d",layersCnt_);
        layer = register_module(name,KANLayer( hidden_dim,output_dim, grid_size, sigma));
        layers_.push_back(layer);


        return;
    }

    torch::Tensor forward(const torch::Tensor& x)
    {
        torch::Tensor h=x;
        for(size_t i=0;i<layersCnt_;i++)
        {
            h = layers_[i]->forward( h);
        }
        return h;
    }
};
TORCH_MODULE(KANNet);

// -----------------------------
// Example usage: fit y = sin(x)
// -----------------------------
int main() {
    torch::manual_seed(0);

    int input_dim = 1;
    int hidden_dim = 64;
    int output_dim = 1;
    int grid_size = 8;
    double sigma = 0.2;
    int n_samples = 512;
    int batch_size=1000;
    auto device = torch::kCPU;
    KANNet model(5,input_dim, hidden_dim, output_dim, grid_size, sigma);
    model->to(device);

    /*auto dataset = DataSrc(batch_size*(n_samples+1), n_samples) .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
                std::move(dataset),
                torch::data::DataLoaderOptions().batch_size(batch_size));
*/
    // create toy dataset: x in [-2π, 2π], y = sin(x)
    auto x = torch::linspace(-2 * M_PI, 2 * M_PI, n_samples).view({n_samples, 1});
    auto y = torch::sin(x);
    x = x.to(device);
    y = y.to(device);

    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(1e-3));
    double dLoss;
    int epochs = 2000;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        model->train();
        optimizer.zero_grad();


        //    auto inputs = batch.data.to(device);   // [batch_size, seq_len, 1]
        //            auto targets = batch.target.to(device); // [batch_size, 1]
        // Reshape to [seq_len, batch_size, input_size]
        //          inputs = inputs.transpose(0, 1); // [seq_len, batch_size, 1]

        optimizer.zero_grad();


        //auto y_pred = model->forward(inputs);
        //auto loss = torch::mse_loss(y_pred, targets);
        auto y_pred = model->forward(x);
        auto loss = torch::mse_loss(y_pred, y);
        dLoss = loss.item<double>();
        loss.backward();      // backprop through KAN
        optimizer.step();     // update parameters

        if (epoch % 200 == 0)
        {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << dLoss << std::endl;
        }
    }

    // quick check on a few points
    model->eval();
    auto x_test = torch::tensor({-3.14, -1.0, 0.0, 1.0, 3.14}).view({5, 1}).to(device);
    auto y_test = torch::sin(x_test);
    auto y_hat = model->forward(x_test);

    std::cout << "\nTest points:\n";
    std::cout << "x    : " << x_test.squeeze() << "\n";
    std::cout << "sin(x): " << y_test.squeeze() << "\n";
    std::cout << "KAN  : " << y_hat.squeeze() << "\n";

    return 0;
}
