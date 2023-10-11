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
struct Options 
{
	int image_size = 224;
	size_t train_batch_size = 8;
	size_t test_batch_size = 200;
	size_t num_workers = 4;
	size_t iterations = 10;
	size_t log_interval = 20;
	// path must end in delimiter
	std::string datasetPath = "/data/imagenet/";
	torch::DeviceType device = torch::kCPU;
};
static Options options;
namespace fs = std::filesystem;
using Data = std::vector<std::pair<cv::Mat, std::string>>;
// using Data = std::vector<cv::Mat>;

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
		if (flipCode == 1) 
			cv::flip(mat, mat, 1); // 1 indicates horizontal flip
	}

	at::Tensor normalize(at::Tensor& tdata) 
	{
		std::vector<double> mean = {0.485, 0.456, 0.406};
		std::vector<double> std = {0.229, 0.224, 0.225};

		tdata = torch::data::transforms::Normalize<>(mean, std)(tdata);
		return tdata;
	}


	Example get(size_t index) 
	{
		auto mat = data[index].first;
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
						.to(torch::kFloat)/255.0;
		


		tdata = normalize(tdata);
		long k = 0;
		auto tlabel = torch::from_blob(&k, {1}, torch::kLong);
		return {tdata, tlabel};
	}

	torch::optional<size_t> size() const 
	{
		return data.size();
	}
};

std::pair<Data, Data> readInfo() 
{
	// HAVE TO FINISH THIS
	Data train, test;
	std::string trainDir = options.datasetPath + "train";
    std::string valDir = options.datasetPath + "val";

    for (const auto& entry : fs::directory_iterator(trainDir)) {
        if (fs::is_directory(entry)) {
            std::string className = entry.path().filename();
            for (const auto& imageEntry : fs::directory_iterator(entry)) {
                if (fs::is_regular_file(imageEntry) && imageEntry.path().extension() == ".JPEG") {
                    cv::Mat image = cv::imread(imageEntry.path().string(), cv::IMREAD_COLOR);
                    if (!image.empty()) {
						train.push_back(std::make_pair(image, className));
                    } else {
                        std::cerr << "Error loading image: " << imageEntry.path().string() << std::endl;
                    }
                }
            }
        }
    }

    // DIR* dir = opendir(trainDir.c_str());
    // struct dirent* entry;
	// if(dir != NULL)
    // while ((entry = readdir(dir)) != nullptr) {
    //     if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
    //         std::string className = entry->d_name;
    //         std::string classPath = trainDir + "/" + className;

    //         DIR* classDir = opendir(classPath.c_str());
    //         struct dirent* imageEntry;

    //         while ((imageEntry = readdir(classDir)) != nullptr) {
    //             if (imageEntry->d_type == DT_REG && strcmp(imageEntry->d_name, ".") != 0 && strcmp(imageEntry->d_name, "..") != 0) {
    //                 std::string imagePath = classPath + "/" + imageEntry->d_name;
    //                 cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    //                 if (!image.empty()) {
	// 					train.push_back(std::make_pair(image, className));
    //                 }
    //             }
    //         }

    //         closedir(classDir);
    //     }
    // }

	// if(dir != NULL)
    // closedir(dir);


    for (const auto& entry : fs::directory_iterator(valDir)) {
        if (fs::is_directory(entry)) {
            std::string className = entry.path().filename();
            for (const auto& imageEntry : fs::directory_iterator(entry)) {
                if (fs::is_regular_file(imageEntry) && imageEntry.path().extension() == ".JPEG") {
                    cv::Mat image = cv::imread(imageEntry.path().string(), cv::IMREAD_COLOR);
                    if (!image.empty()) {
						test.push_back(std::make_pair(image, className));
                    } else {
                        std::cerr << "Error loading image: " << imageEntry.path().string() << std::endl;
                    }
                }
            }
        }
    }

    // dir = opendir(valDir.c_str());

	// if(dir != NULL)
    // while ((entry = readdir(dir)) != nullptr) {
    //     if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
    //         std::string className = entry->d_name;
    //         std::string classPath = valDir + "/" + className;

    //         DIR* classDir = opendir(classPath.c_str());
    //         struct dirent* imageEntry;

    //         while ((imageEntry = readdir(classDir)) != nullptr) {
    //             if (imageEntry->d_type == DT_REG && strcmp(imageEntry->d_name, ".") != 0 && strcmp(imageEntry->d_name, "..") != 0) {
    //                 std::string imagePath = classPath + "/" + imageEntry->d_name;
    //                 cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    //                 if (!image.empty()) {
	// 					test.push_back(std::make_pair(image, className));
    //                 }
    //             }
    //         }

    //         closedir(classDir);
    //     }
    // }

	// if(dir != NULL)
    // closedir(dir);

	return std::make_pair(train, test);
}




// TORCH_MODULE(Network);
template <typename DataLoader>
void train(DataLoader& loader, torch::jit::script::Module& model, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size) 
{
	size_t index = 0;
	model.train();
	float Loss = 0, Acc = 0;

	for (auto& batch : loader) {
		auto data = batch.data.to(options.device);
		auto targets = batch.target.to(options.device).view({-1});

		std::vector<c10::IValue>temp_op;

		temp_op.push_back(data);

		auto output = model.forward(temp_op).toTensor();
		auto loss = torch::nll_loss(output, targets);
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

template <typename DataLoader>
void test(DataLoader& loader, torch::jit::script::Module& model, size_t data_size) 
{
	size_t index = 0;
	model.eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	for (const auto& batch : loader) 
	{
		auto data = batch.data.to(options.device);
		auto targets = batch.target.to(options.device).view({-1});

		std::vector<c10::IValue>temp_op;

		temp_op.push_back(data);

		auto output = model.forward(temp_op).toTensor();
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

int main(int argc, const char* argv[]) 
{
	if (argc != 2) 
	{
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}
  
	torch::jit::script::Module model;
	try 
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().
		model = torch::jit::load(argv[1]);
	}
	catch (const c10::Error& e) 
	{
		std::cerr << "error loading the model\n";
		return -1;
	}

  	std::cout << "ok\n";
	if (torch::cuda::is_available())
		options.device = torch::kCUDA;
	
	auto data = readInfo();

	auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
	auto train_size = train_set.size().value();
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_set), torch::data::DataLoaderOptions().batch_size(options.train_batch_size).workers(options.num_workers));

	auto test_set = CustomDataset(data.second).map(torch::data::transforms::Stack<>());
	auto test_size = test_set.size().value();
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_set), torch::data::DataLoaderOptions().batch_size(options.test_batch_size).workers(options.num_workers));

	model.to(options.device);

	std::cout << "Model Parameters: " << model.parameters().size() << std::endl;

	torch::jit::parameter_list parameters = model.parameters();

	std::vector<torch::Tensor> tensor_vector;

	for (const torch::jit::IValue& param : parameters) {
		if (param.isTensor()) {
			torch::Tensor tensor = param.toTensor();
			tensor_vector.push_back(tensor);
		}
	}

	torch::optim::SGD optimizer(tensor_vector, torch::optim::SGDOptions(0.001).momentum(0.5));

	for (size_t i = 0; i < options.iterations; ++i) {
		train(*train_loader, model, optimizer, i + 1, train_size);
		std::cout << std::endl;
		test(*test_loader, model, test_size);
		std::cout << std::endl;
	}

	return 0;
}

