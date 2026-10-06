#include "DataSrc.h"

void CreateInput(int64_t data_size,std::vector<float>& raw_data)
{

    for(int i=0;i<data_size;i++)
    {
        raw_data.push_back( std::sin((M_PI/100.0) * i));
    }

}


DataSrc::DataSrc( int64_t full_seq_len, int64_t  seq_len)
{
    std::vector<float> raw_data;

    CreateInput(full_seq_len,raw_data);

    uint64_t size=full_seq_len/(seq_len + 1);
    auto startPoint  = raw_data.begin();

    for(uint64_t i=0;i<size;i++)
    {
         std::vector<float> seq(startPoint + i, startPoint + i + seq_len);
        auto input_tensor = torch::tensor(seq).view({seq_len, 1});   // [seq_len, input_dim=1]
        auto target_tensor = torch::tensor({*(startPoint + i + seq_len)}).view({1});  // [1]

        inputs_.push_back(input_tensor);
        targets_.push_back(target_tensor);


        ++startPoint;
    }
}
