#ifndef  _DATA_SRC_
#define  _DATA_SRC_

#include <torch/torch.h>
#include <iostream>
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


using namespace std;

struct DataSrc : torch::data::Dataset<DataSrc> {
    std::vector<torch::Tensor> inputs_;
    std::vector<torch::Tensor> targets_;

    explicit DataSrc(  int64_t full_seq_len, int64_t  seq_len);

    torch::data::Example<> get(size_t index) override
    {
        return {inputs_.at(index), targets_.at(index)};
    }

    torch::optional<size_t> size() const override {
        return inputs_.size();
    }
};


#endif
