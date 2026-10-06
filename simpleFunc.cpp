#include "simpleFunc.h"
#include "simpleNet.h"

const char* kDataRoot = "/home/leonidp/torchSmpl/data/";
const int64_t kTrainBatchSize = 64;
const int64_t kTestBatchSize = 100;
const int64_t kLogInterval = 10;

template <typename DataLoader>
float train(size_t epoch, SimpleNet& model, torch::Device device,
            DataLoader& loader, torch::optim::Optimizer& optimizer,
            size_t dataset_size)
{
    model.train();
    torch::Tensor loss ;

    size_t batch_idx = 0;

    for (auto& batch : loader)
    {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model.forward(data);
        loss = torch::nll_loss(output, targets);
        loss.backward();
        optimizer.step();

        /*  if (batch_idx++ % kLogInterval == 0)
        {
            std::cout << "Train Epoch: " << epoch
                      << " [" << batch_idx * batch.data.size(0)
                      << "/" << dataset_size << "] Loss: "
                      << loss.item().toSymFloat().expect_float() << std::endl;
        }*/
    }
    return loss.item().toSymFloat().expect_float();
}

template <typename DataLoader>
bool  test(SimpleNet& model, torch::Device device,
           DataLoader& loader, size_t dataset_size,double expected)
{
    model.eval();
    torch::NoGradGuard no_grad;
    double correct = 0;

    for (const auto& batch : loader)
    {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model.forward(data);
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().item().toSymFloat().expect_float();
    }

    /*std::cout << "Test Accuracy: "
              << (100.0 * correct / dataset_size) << "%" << std::endl;*/

    return (100.0 * correct / dataset_size) > expected;
}

nlohmann::json SimpleFunc(nlohmann::json params)
{
    torch::manual_seed(1);
    torch::Device device( torch::kCPU);

    SimpleNet model;
    model.to(device);
    model.Load();
    auto train_dataset = torch::data::datasets::MNIST(kDataRoot)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
                train_dataset, kTrainBatchSize);

    auto test_dataset = torch::data::datasets::MNIST(kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader(
                test_dataset, kTestBatchSize);


    auto options = torch::optim::AdamOptions(1e-3)
            .betas(std::make_tuple(0.9, 0.999))
            .eps(1e-8)
            .weight_decay(0.0001);

    // 2. Instantiate the optimizer with model parameters and options
    torch::optim::Adam optimizer(model.parameters(), options);
    /***********************************************************************************/
    float loss =1e9;
    size_t epoch;



    for ( epoch = 1; epoch <= params["maxEpoch"].get<size_t>(); ++epoch)
    {
        loss = train(epoch, model, device, *train_loader, optimizer, train_dataset.size().value());
        if(test(model, device, *test_loader, test_dataset.size().value(),99))
        {
            break;
        }
    }

    //model.Save();

    std::string explanation=
            /* "layers-" + std::to_string(params["layers"].get<int>() )+
                    " InternalSize - " + std::to_string(params["intSize"].get<int>())  +*/
            " Real Epochs - " +std::to_string(epoch) ;

    params["explanation"] =explanation;
    params["loss_result"] =loss;

    return params;
}

