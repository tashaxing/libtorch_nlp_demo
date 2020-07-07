#include <iostream>
#include <string>
#include <vector>
#include "torch/script.h"

int main(int argc, char* argv[])
{
	torch::jit::script::Module net;

	try 
	{
		// read model from file
		std::string model_file_path = "text_rnn.pt";
		net = torch::jit::load(model_file_path); 

		// optional: set device
		torch::DeviceType device_type = torch::kCPU; // default run on cpu, you may use kCUDA
		torch::Device device(device_type, 0);
		net.to(device);

		// create inputs, watch out that a::Tensor and torch::Tensor is the same type
		// sentence1: "I feel the movie to kind of great and to my taste"
		torch::Tensor input1 = torch::tensor({ {9, 223, 2, 20, 232, 5, 88, 4, 6, 57, 1743} }); // adapt the shape as a batch fo samples
		torch::Tensor output1 = net.forward({ input1 }).toTensor();
		std::cout << "output1: " << output1 << std::endl;

		// sentence2: "the movie has bad experience"
		torch::Tensor  input2 = torch::tensor({ {2,  20,  41,  97, 802} }); // adapt the shape as a batch fo samples
		std::vector<torch::jit::IValue> inputs{ input2 };
		torch::Tensor output2 = net.forward(inputs).toTensor();
		std::cout << "output2: " << output2 << std::endl;

		// TODO: argmax

		std::cout << "predict ok" << std::endl;
		
	}
	catch (const c10::Error& e) 
	{
		std::cerr << "error loading the model, error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}
