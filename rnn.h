#include <torch/torch.h>
#include <iostream>
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>



struct RNNModelImpl : torch::nn::Module {
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};

    RNNModelImpl(int64_t input_size, int64_t hidden_size, int64_t num_layers, int64_t output_size) {
        lstm = register_module("lstm",
                               torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size).num_layers(num_layers)));
        fc = register_module("fc", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [seq_len, batch_size, input_size]
        auto lstm_out = lstm->forward(x);
        auto output_seq = std::get<0>(lstm_out); // [seq_len, batch_size, hidden_size]

        // Use last time step
        auto last_step = output_seq[-1]; // [batch_size, hidden_size]

        auto out = fc->forward(last_step); // [batch_size, output_size]
        return out;
    }
};
TORCH_MODULE(RNNModel);


float runRNN(int64_t hidden_size,int64_t num_layers,uint32_t maxEpoch,uint32_t& realEpoch);
