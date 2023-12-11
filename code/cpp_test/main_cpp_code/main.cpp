#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <map>
#include "resnet.h"

#define LOG__(x) std::cout << x << std::endl

struct Options 
{
	int image_size = 224;
	size_t train_batch_size = 256;
	size_t test_batch_size = 256;
	size_t num_workers = 4;
	size_t iterations = 10;
	size_t log_interval = 20;
	// path must end in delimiter
	std::string datasetPath = "/data/imagenet/";
	torch::DeviceType device = torch::kCPU;
};
static Options options;
namespace fs = std::filesystem;
using Data = std::vector<std::pair<std::string, long>>;
typedef std::vector<std::string> StringList;
typedef std::map<std::string, int> StringToIntMap;
// using Data = std::vector<cv::Mat>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
	using Example = torch::data::Example<>;

	Data data;

	public:
	CustomDataset(const Data& data) : data(data) 
	{}

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
		int x = (mat.size().width - maxCropWidth) > 0 ? rand() % (mat.size().width - maxCropWidth) : 0;
		int y = (mat.size().height - maxCropHeight) > 0 ? rand() % (mat.size().height - maxCropHeight) : 0;

		// Ensure the crop box is within the image bounds
		x = std::max(0, x);
		y = std::max(0, y);
		int x2 = std::min(maxCropWidth, mat.size().width - x);
		int y2 = std::min(maxCropWidth, mat.size().height - y);

		// Extract the random crop from the input image, considering scale and aspect ratio
		cv::Rect cropRect(x, y, x2, y2);
		mat = mat(cropRect);

		// Resize the cropped image to the target size (e.g., 224x224)
		cv::resize(mat, mat, cv::Size(options.image_size, options.image_size), 0, 0, 1);
	}

	void randomHorizontalFlip(cv::Mat &mat) 
	{
		int flipCode = rand() % 2;

		// Apply horizontal flip if flipCode is 1
		if (flipCode == 1) 
			cv::flip(mat, mat, 1); // 1 indicates horizontal flip
	}

	void normalize(at::Tensor& tdata) 
	{
		std::vector<double> mean = {0.485, 0.456, 0.406};
		std::vector<double> std = {0.229, 0.224, 0.225};

		tdata = torch::data::transforms::Normalize<>(mean, std)(tdata);
	}


	Example get(size_t index) 
	{
		static std::pair<std::chrono::duration<double>, std::chrono::duration<double>> elapsed;
		static size_t count = 0;
		std::string path = data[index].first;
		auto mat = cv::imread(path);
		assert(!mat.empty());

		// Time a function call and execution
		// std::chrono::high_resolution_clock represents the clock with the smallest tick period
		auto start = std::chrono::high_resolution_clock::now();
		randomResizedCrop(mat);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_rrc = end - start;
		elapsed.first += elapsed_rrc;
		// LOG__("Time taken by randomResizedCrop: " << elapsed_rrc.count() << " seconds");

		start = std::chrono::high_resolution_clock::now();
		randomHorizontalFlip(mat);
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_rhf = end - start;
		elapsed.second += elapsed_rhf;
		// LOG__("Time taken by randomHorizontalFlip: " << elapsed_rhf.count() << " seconds");
		
		if (count++ % 256 == 0)
		{
			LOG__("Time taken by randomResizedCrop: " << elapsed.first.count() << " seconds and randomHorizontalFlip: " << elapsed.second.count() << " seconds for " << count << " images");
			LOG__("Average time taken by randomResizedCrop: " << elapsed.first.count() / count << " seconds and randomHorizontalFlip: " << elapsed.second.count() / count << " seconds given " << count << " images");
		}

		// std::vector<cv::Mat> channels(3);
		// cv::split(mat, channels);
		// auto R = torch::from_blob(
		// 	channels[2].ptr(),
		// 	{options.image_size, options.image_size},
		// 	torch::kUInt8);
		// auto G = torch::from_blob(
		// 	channels[1].ptr(),
		// 	{options.image_size, options.image_size},
		// 	torch::kUInt8);
		// auto B = torch::from_blob(
		// 	channels[0].ptr(),
		// 	{options.image_size, options.image_size},
		// 	torch::kUInt8);
		// auto tdata = torch::cat({R, G, B})
		// 				.view({3, options.image_size, options.image_size})
		// 				.to(torch::kFloat).div_(255);
		auto tdata = torch::from_blob(mat.data, {options.image_size, options.image_size, 3}, at::kByte).permute({2, 0, 1}).to(torch::kFloat).div_(255);
		// auto tdata = torch::from_blob(mat.data, {options.image_size, options.image_size, 3}, torch::kUInt8).permute({2, 0, 1}).to(torch::kFloat).div_(255);
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
	Data train, val;
	std::string trainDir = options.datasetPath + "train";
    std::string valDir = options.datasetPath + "val";
	
	// Generate labels for train data
	StringList trainClasses;
    StringToIntMap trainClassToIdx;
    for (const auto& entry : fs::directory_iterator(trainDir)) 
	{
        if (entry.is_directory()) 
		{
            trainClasses.push_back(entry.path().filename());
        }
    }
    std::sort(trainClasses.begin(), trainClasses.end());
    if (trainClasses.empty()) 
	{
        throw std::runtime_error("Couldn't find any class folder in " + trainDir);
    }
    long index = 0;
    for (const std::string& cls : trainClasses) 
	{
        trainClassToIdx[cls] = index;
        index++;
    }

	// Load train data
    for (const auto& entry : fs::directory_iterator(trainDir))
	{
        if (fs::is_directory(entry))
		{
            std::string className = entry.path().filename();
            for (const auto& imageEntry : fs::directory_iterator(entry)) 
			{
                if (fs::is_regular_file(imageEntry) && imageEntry.path().extension() == ".JPEG") 
				{
					train.push_back(std::make_pair(imageEntry.path().string(), trainClassToIdx[className]));
                }
            }
        }
    }

	// Generate labels for val data
	StringList valClasses;
    StringToIntMap valClassToIdx;
    for (const auto& entry : fs::directory_iterator(valDir)) 
	{
        if (entry.is_directory() && entry.path().filename() != "ILSVRC2012_img_val") 
		{
            valClasses.push_back(entry.path().filename());
        }
    }
    std::sort(valClasses.begin(), valClasses.end());
    if (valClasses.empty()) 
	{
        throw std::runtime_error("Couldn't find any class folder in " + valDir);
    }
    index = 0;
    for (const std::string& cls : valClasses) 
	{
        valClassToIdx[cls] = index;
        index++;
    }

	// load val data
    for (const auto& entry : fs::directory_iterator(valDir)) 
	{
        if (fs::is_directory(entry) && entry.path().filename() != "ILSVRC2012_img_val") 
		{
            std::string className = entry.path().filename();
            for (const auto& imageEntry : fs::directory_iterator(entry)) 
			{
                if (fs::is_regular_file(imageEntry) && imageEntry.path().extension() == ".JPEG") 
				{
					val.push_back(std::make_pair(imageEntry.path().string(), valClassToIdx[className]));
                }
            }
        }
    }
	return std::make_pair(train, val);
}




// TORCH_MODULE(Network);
template <typename DataLoader, typename Model>
// std::shared_ptr<Model>
// train_model(std::shared_ptr<Model> model, DataLoader& train_data_loader,
// 			DataLoader& test_data_loader, torch::Device device,
// 			float learning_rate, int64_t num_epochs) 
// {
// 	model->to(device);

// 	torch::optim::SGD optimizer(model->parameters(),
// 								torch::optim::SGDOptions(learning_rate));

// 	for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) 
// 	{
// 		std::cout << "Epoch: " << epoch << std::endl;

// 		model->train();
// 		size_t batch_index = 0;
// 		for (auto& batch : *train_data_loader) 
// 		{
// 			auto data = batch.data.to(device), targets = batch.target.to(device);

// 			optimizer.zero_grad();
// 			auto output = model->forward(data);
// 			auto loss = torch::nll_loss(output, targets);
// 			AT_ASSERT(!std::isnan(loss.template item<float>()));
// 			loss.backward();
// 			optimizer.step();

// 			if (batch_index++ % 10 == 0) 
// 			{
// 				std::cout << "Train Loss: " << loss.template item<float>()
// 						  << std::endl;
// 			}
// 		}

// 		model->eval();
// 		torch::NoGradGuard no_grad;
// 		size_t num_correct = 0;
// 		float test_loss = 0;
// 		for (const auto& batch : *test_data_loader) 
// 		{
// 			auto data = batch.data.to(device), targets = batch.target.to(device);

// 			auto output = model->forward(data);
// 			test_loss += torch::nll_loss(
// 							 output,
// 							 targets,
// 							 /*weight=*/{},
// 							 torch::Reduction::Sum)
// 							 .template item<float>();
// 			auto pred = output.argmax(1);
// 			num_correct += pred.eq(targets).sum().template item<int64_t>();
// 		}

// 		test_loss /= test_data_loader->size().value();
// 		std::cout << "Test Loss: " << test_loss
// 				  << ", Accuracy: " << static_cast<float>(num_correct)
// 						 / test_data_loader->size().value()
// 				  << std::endl;
// 	}

// 	return model;
// }
// void train(DataLoader& loader, torch::jit::script::Module& model, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size) 
void train(DataLoader& loader, std::shared_ptr<Model> model, torch::optim::SGD& optimizer, size_t epoch, size_t data_size)
{
	size_t index = 0;
	model->train();
	float Loss = 0, Acc = 0;
	for (auto& batch : loader) 
	{
		auto data = batch.data.to(options.device);
		auto targets = batch.target.to(options.device).view({-1});

		// Time taken to process a batch
		auto start = std::chrono::high_resolution_clock::now();
		auto output = model->forward(data);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		LOG__("Time taken by forward pass: " << elapsed.count() << " seconds for batch with size: " << data.size(0));

		auto loss = torch::cross_entropy_loss(output, targets).to(options.device);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();
		
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		if (index++ % options.log_interval == 0) 
		{
			auto end = std::min(data_size, (index + 1) * options.train_batch_size);

			std::cout << "Train Epoch: " << epoch << " " << end << "/" << data_size
						<< "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
						<< std::endl;
		}
	}
}

template <typename DataLoader, typename Model>
void test(DataLoader& loader, std::shared_ptr<Model> model, size_t data_size) 
{
	size_t index = 0;
	model->eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	for (const auto& batch : loader) 
	{
		auto data = batch.data.to(options.device);
		auto targets = batch.target.to(options.device).view({-1});

		auto output = model->forward(data);
		auto loss = torch::cross_entropy_loss(output, targets).to(options.device);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}

	if (index++ % options.log_interval == 0)
		std::cout << "Test Loss: " << Loss / data_size
				<< "\tAcc: " << Acc / data_size << std::endl;
}

int main(int argc, const char* argv[]) 
{

	std::shared_ptr<ResNet<BasicBlock>> model = resnet18(/*num_classes = */ 1000);

	if (torch::cuda::is_available())
		options.device = torch::kCUDA;
  	std::cout << "Running on: " << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
	
	auto data = readInfo();

	// auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
	auto train_set = CustomDataset(data.first)
	.map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
	.map(torch::data::transforms::Stack<>());
	auto train_size = train_set.size().value();
	auto train_loader = torch::data::make_data_loader(std::move(train_set), torch::data::DataLoaderOptions().batch_size(options.train_batch_size));//.workers(options.num_workers));

	auto test_set = CustomDataset(data.second)
	.map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}))
	.map(torch::data::transforms::Stack<>());
	auto test_size = test_set.size().value();
	auto test_loader = torch::data::make_data_loader(std::move(test_set), torch::data::DataLoaderOptions().batch_size(options.test_batch_size));//.workers(options.num_workers));

	model->to(options.device);

	// lr = 0.1 weightDecay = 0.0001 momentum = 0.9
	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).weight_decay(0.0001).momentum(0.9));
	
	for (size_t i = 0; i < options.iterations; ++i) 
	{
		train(*train_loader, model, optimizer, i + 1, train_size);
		std::cout << std::endl;
		test(*test_loader, model, test_size);
		std::cout << std::endl;
	}

	return 0;
}

