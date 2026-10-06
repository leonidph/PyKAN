#include "rnn.h"
#include "DataSrc.h"

float runRNN(int64_t hidden_size,int64_t num_layers,uint32_t maxEpoch,uint32_t& realEpoch)
{
    // Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
   // std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    // Hyperparameters
    const int64_t input_size   = 1;
    const int64_t output_size  = 1;
    const int64_t seq_len      = 20;
    size_t  batch_size   = 1600;
    const double  learning_rate = 0.01;

    const double  exp_loss = 0.00001;



    // Dataset & DataLoader (your own data source)
    //const std::string csv_path = "/home/leonidp/RNNPyTorch/data.csv"; // change to your path
    //auto dataset = FileSequenceDataset(csv_path, seq_len) .map(torch::data::transforms::Stack<>());

    std::string stock = "/home/leonidp/.cache/kagglehub/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/versions/3/Stocks/zion.us.txt";

    //auto dataset = StockData(stock,3,batch_size, seq_len) .map(torch::data::transforms::Stack<>());
    auto dataset = DataSrc(batch_size*(seq_len+1), seq_len) .map(torch::data::transforms::Stack<>());


    auto data_loader = torch::data::make_data_loader(
                std::move(dataset),
                torch::data::DataLoaderOptions().batch_size(batch_size)); //

    // Model
    RNNModel model(input_size, hidden_size, num_layers, output_size);
    model->to(device);

    // Optimizer & loss
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    auto criterion = torch::nn::MSELoss();
    float loss=100;
    // Training loop
    for (realEpoch = 0; realEpoch < maxEpoch; ++realEpoch) {
        double epoch_loss = 0.0;
        size_t batch_idx = 0;

        for (auto& batch : *data_loader) {
            auto inputs = batch.data.to(device);   // [batch_size, seq_len, 1]
            auto targets = batch.target.to(device); // [batch_size, 1]

            // Reshape to [seq_len, batch_size, input_size]
            inputs = inputs.transpose(0, 1); // [seq_len, batch_size, 1]

            optimizer.zero_grad();

            auto outputs = model->forward(inputs); // [batch_size, 1]

            auto loss = criterion(outputs, targets);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            batch_idx++;
        }

        loss = epoch_loss / batch_idx;
        if(loss <= exp_loss)
        {
            break;
        }
    }


    return loss;
}


