#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Options {
  int image_size = 224;
  size_t train_batch_size = 8;
  size_t test_batch_size = 200;
  size_t iterations = 10;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string datasetPath = "/data/mrao70/datasets/caltech-101/101_ObjectCategories/";
  std::string infoFilePath = "/data/mrao70/datasets/info.txt";
  torch::DeviceType device = torch::kCPU;
};

static Options options;
namespace fs = std::filesystem;
using Data = std::vector<std::pair<std::string, long>>;


// https://stackoverflow.com/questions/63502473/different-output-from-libtorch-c-and-pytorch
// https://discuss.pytorch.org/t/libtorch-c-equivalent-of-torch-transforms-compose-function-please-help/176605/2

auto randomHorizontalFlip = [](torch::data::Example<> input) -> torch::data::Example<> {
    int flipCode = rand() % 2;

    if (flipCode == 1) {
        // Apply horizontal flip to the tensor along the width dimension (axis 2)
        input.data = torch::flip(input.data, {2});
    }

    return input;
};

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
  using Example = torch::data::Example<>;

  Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  void randomResizedCrop(cv::Mat &mat) 
  {
    float minScale = 0.8;  // Minimum scaling factor
    // float minScale = 1.0;  // Minimum scaling factor
    float maxScale = 1.0;  // Maximum scaling factor

    // A range of aspect ratios to randomly select from
    float minAspectRatio = 0.75;
    // float minAspectRatio = 1.0;
    float maxAspectRatio = 1.3333;
    // float maxAspectRatio = 1.0;

    // By applying the scale factor to the width and the aspect ratio factor to the height, 
    // we effectively control both the size and aspect ratio of the cropped region, 
    // allowing us to meet both scaling and aspect ratio requirements simultaneously.
    // Calculate the width and height of the crop box based on the aspect ratio
    int maxCropWidth, maxCropHeight;
    try
    {
      // Calculate the maximum possible crop dimensions based on scale and aspect ratio
      float randomScale = minScale + static_cast<float>(rand()) / RAND_MAX * (maxScale - minScale);
      float randomAspectRatio = minAspectRatio + static_cast<float>(rand()) / RAND_MAX * (maxAspectRatio - minAspectRatio);

      maxCropWidth = (int)(mat.size().width * randomScale);
      maxCropHeight = (int)(maxCropWidth / randomAspectRatio);
    }
    catch(const std::exception& e)
    {
      std::cerr << e.what() << '\n';
    }
    
    // Generate random coordinates for the top-left corner of the crop box
    int x = mat.size().width - maxCropWidth ? rand() % (mat.size().width - maxCropWidth) : 0;
    int y = mat.size().height - maxCropHeight ? rand() % (mat.size().height - maxCropHeight) : 0;

    // Ensure the crop box is within the image bounds
    x = std::max(0, x);
    y = std::max(0, y);
    int x2 = std::min(maxCropWidth, mat.size().width - x);
    int y2 = std::min(maxCropWidth, mat.size().height - y);

    // Extract the random crop from the input image, considering scale and aspect ratio
    cv::Rect cropRect(x, y, x2, y2);
    mat = mat(cropRect);

    // Resize the cropped image to the target size (e.g., 224x224)
    cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
  }

  void randomHorizontalFlip(cv::Mat &mat) {
    int flipCode = rand() % 2;

    // Apply horizontal flip if flipCode is 1
    if (flipCode == 1) {
        cv::flip(mat, mat, 1); // 1 indicates horizontal flip
    }
  }

  void normalize(at::Tensor& tdata) 
  {
    std::vector<double> mean = {0.485, 0.456, 0.406};
    std::vector<double> std = {0.229, 0.224, 0.225};
    tdata = torch::data::transforms::Normalize<>(mean, std)(tdata);
  }


  Example get(size_t index) 
  {
    std::string path = options.datasetPath + data[index].first;
    auto mat = cv::imread(path);
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
                     .to(torch::kFloat);
    
    normalize(tdata);
    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
    return {tdata, tlabel};
  }

  torch::optional<size_t> size() const 
  {
    return data.size();
  }
};

std::pair<Data, Data> readInfo() 
{
  Data train, test;

  std::ifstream stream(options.infoFilePath);
  assert(stream.is_open());

  long label;
  std::string path, type;

  while (true) {
    stream >> path >> label >> type;

    if (type == "train")
      train.push_back(std::make_pair(path, label));
    else if (type == "test")
      test.push_back(std::make_pair(path, label));
    else
      assert(false);

    if (stream.eof())
      break;
  }

  std::random_shuffle(train.begin(), train.end());
  std::random_shuffle(test.begin(), test.end());
  return std::make_pair(train, test);
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

  //if (torch::cuda::is_available())
  //  options.device = torch::kCUDA;
  std::cout << "Running on: "
            << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto data = readInfo();

  // auto train_set = CustomDataset(data.first).map(torch::data::transforms::Lambda<torch::data::Example<>>(randomHorizontalFlip)).map(torch::data::transforms::Stack<>());
  auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
  auto train_size = train_set.size().value();
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_set), 10);

  auto test_set = CustomDataset(data.second).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();
  auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), options.test_batch_size);

  Network network;
  network->to(options.device);

  torch::optim::SGD optimizer(
      network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

  for (size_t i = 0; i < options.iterations; ++i) {
    train(network, *train_loader, optimizer, i + 1, train_size);
    std::cout << std::endl;
    test(network, *test_loader, test_size);
    std::cout << std::endl;
  }

  return 0;
}
