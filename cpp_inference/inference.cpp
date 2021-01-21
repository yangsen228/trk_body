#include <torch/script.h> // One-stop header.
#include <iostream>
#include <ctime>
#include <memory>

int main() {
	// Deserialize the ScriptModule from a file using torch::jit::load().
	torch::jit::script::Module module = torch::jit::load("../models/model_parameters_nn_64_100.pt");
    // torch::jit::script::Module module = torch::jit::load("../models/20201203_norm_with_aug_256_ch512_rf9.pt");
    module.to(at::kCUDA);
 
	std::cout << "ok\n";

    int N = 10000;
    clock_t startTime,endTime;
    startTime = clock();
    for (int i = 0; i < N; i++) {
        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::ones({ 1, 90 }).to(at::kCUDA));
        // inputs.push_back(torch::ones({ 1, 9, 15, 3 }).to(at::kCUDA));
    
        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();
        // std::cout << output.slice(1, 0, 5) << '\n';
    }
    endTime = clock();
    std::cout << "The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC / (double)N << "s" << std::endl;

	system("pause");
	return 0;
}