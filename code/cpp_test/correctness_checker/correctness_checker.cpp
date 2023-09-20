#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Options {
  int image_size = 64;
  size_t train_batch_size = 8;
  size_t test_batch_size = 1;
  size_t iterations = 10;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string datasetPath = "./imagenet/";
  std::string infoFilePath = "info.txt";
  torch::DeviceType device = torch::kCPU;
};

static Options options;
namespace fs = std::filesystem;
using Data = std::vector<cv::Mat>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
  using Example = torch::data::Example<>;

  Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  void randomResizedCrop(cv::Mat &mat) 
  {
    float minScale = 1.0;  // Minimum scaling factor
    float maxScale = 1.0;  // Maximum scaling factor

    float minAspectRatio = 1.0;
    float maxAspectRatio = 1.0;

    int maxCropWidth, maxCropHeight;
    try
    {
      float randomScale = 1.0;
      float randomAspectRatio = 1.0;

      maxCropWidth = (int)(mat.size().width * randomScale);
      maxCropHeight = (int)(maxCropWidth / randomAspectRatio);
    }
    catch(const std::exception& e)
    {
      std::cerr << e.what() << '\n';
    }
    
    int x = mat.size().width - maxCropWidth ? rand() % (mat.size().width - maxCropWidth) : 0;
    int y = mat.size().height - maxCropHeight ? rand() % (mat.size().height - maxCropHeight) : 0;

    x = std::max(0, x);
    y = std::max(0, y);
    int x2 = std::min(maxCropWidth, mat.size().width - x);
    int y2 = std::min(maxCropWidth, mat.size().height - y);

    cv::Rect cropRect(x, y, x2, y2);
    mat = mat(cropRect);

    cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
  }

  void randomHorizontalFlip(cv::Mat &mat) {
    int flipCode = 1;

    if (flipCode == 1) {
        cv::flip(mat, mat, 1); // 1 indicates horizontal flip
    }
  }

  at::Tensor normalize(at::Tensor& tdata) 
  {
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};

    tdata = torch::data::transforms::Normalize<>(mean, std)(tdata);
    return tdata;
  }

  Example get(size_t index) {

    auto mat = data[index];
    assert(!mat.empty());
    randomResizedCrop(mat);
    randomHorizontalFlip(mat);
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);
    auto R = torch::from_blob(
        channels[2].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto G = torch::from_blob(
        channels[1].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto B = torch::from_blob(
        channels[0].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto tdata = torch::cat({R, G, B})
                     .view({3, options.image_size, options.image_size})
                     .to(torch::kDouble)/255.0;
    


    tdata = normalize(tdata);
    long k = 0;
    auto tlabel = torch::from_blob(&k, {1}, torch::kLong);
    return {tdata, tlabel};
  }

  torch::optional<size_t> size() const {
    return data.size();
  }
};

Data readInfo() 
{
  Data train, test;

  try 
  {
    for (const auto& entry : fs::directory_iterator(options.datasetPath + "test/images2/")) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            cv::Mat image = cv::imread(filePath);

            if (image.empty()) {
                std::cerr << "Error reading image: " << filePath << std::endl;
            } else {
              test.push_back(image);
            }
        }
    }
  } catch (const fs::filesystem_error& ex) {
      std::cerr << "Error: " << ex.what() << std::endl;
  }
  return test;
}

struct NetworkImpl : torch::nn::SequentialImpl {
  NetworkImpl() {
    using namespace torch::nn;

    auto stride = torch::ExpandingArray<2>({2, 2});
    torch::ExpandingArray<2> shape({-1, 256 * 6 * 6});
    push_back(Conv2d(Conv2dOptions(3, 64, 11).stride(4).padding(2)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(64, 192, 5).padding(2)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Conv2d(Conv2dOptions(192, 384, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Conv2d(Conv2dOptions(384, 256, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Conv2d(Conv2dOptions(256, 256, 3).padding(1)));
    push_back(Functional(torch::relu));
    push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
    push_back(Functional(torch::reshape, shape));
    push_back(Dropout());
    push_back(Linear(256 * 6 * 6, 4096));
    push_back(Functional(torch::relu));
    push_back(Dropout());
    push_back(Linear(4096, 4096));
    push_back(Functional(torch::relu));
    push_back(Linear(4096, 102));
    // push_back(Functional(torch::log_softmax, 1, torch::nullopt));
    push_back(Functional(static_cast<torch::Tensor(&)(const torch::Tensor&, int64_t, torch::optional<torch::ScalarType> )>(torch::log_softmax), 1, torch::nullopt));
  }
};
TORCH_MODULE(Network);

template <typename DataLoader>
void train(
    Network& network,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t data_size) {
  size_t index = 0;
  network->train();
  float Loss = 0, Acc = 0;

  for (auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();

    if (index++ % options.log_interval == 0) {
      auto end = std::min(data_size, (index + 1) * options.train_batch_size);

      std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
                << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
                << std::endl;
    }
  }
}

template <typename DataLoader>
void test(Network& network, DataLoader& loader, size_t data_size) {
  size_t index = 0;
  network->eval();
  torch::NoGradGuard no_grad;
  float Loss = 0, Acc = 0;

  for (const auto& batch : loader) {
    auto data = batch.data.to(options.device);
    auto targets = batch.target.to(options.device).view({-1});

    auto output = network->forward(data);
    auto loss = torch::nll_loss(output, targets);
    assert(!std::isnan(loss.template item<float>()));
    auto acc = output.argmax(1).eq(targets).sum();

    Loss += loss.template item<float>();
    Acc += acc.template item<float>();
  }

  if (index++ % options.log_interval == 0)
    std::cout << "Test Loss: " << Loss / data_size
              << "\tAcc: " << Acc / data_size << std::endl;
}

int main() {
  torch::manual_seed(1);

  if (torch::cuda::is_available())
    options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto data = readInfo();

  // auto train_set = CustomDataset(data.first).map(torch::data::transforms::Lambda<torch::data::Example<>>(randomHorizontalFlip)).map(torch::data::transforms::Stack<>());
  // auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
  // // std::cout << train_set << std::endl;
  // auto train_size = train_set.size().value();
  // auto train_loader =
  //     torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
  //         std::move(train_set), 10);


  // auto data_loader = torch::data::make_data_loader(
  //     std::move(dataset),
  //     torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  auto test_set = CustomDataset(data).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), options.test_batch_size);

  std::ofstream outputFile("output.txt");
    
  for (auto& batch : *test_loader) 
  {
    auto data = batch.data.to(options.device);
    outputFile << data << std::endl;
    break;
  }
  
  // return 0;

  // Network network;
  // network->to(options.device);

  // torch::optim::SGD optimizer(
  //     network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

  // for (size_t i = 0; i < options.iterations; ++i) {
  //   train(network, *train_loader, optimizer, i + 1, train_size);
  //   std::cout << std::endl;
  //   test(network, *test_loader, test_size);
  //   std::cout << std::endl;
  // }

  return 0;
}
