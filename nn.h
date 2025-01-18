// nn.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <random>

using namespace std;

class NeuralNetwork {
private:
    vector<vector<double>> weights_hidden;
    vector<double> bias_hidden;
    vector<vector<double>> weights_output;
    vector<double> bias_output;
    double learning_rate;
    // 初始化批量梯度
    vector<vector<double>> batch_weights_hidden_grad;
    vector<double> batch_bias_hidden_grad;
    vector<vector<double>> batch_weights_output_grad;
    vector<double> batch_bias_output_grad;

    // 激活函数和导数
    function<vector<double>(const vector<double>&)> activation_function;
    function<vector<double>(const vector<double>&)> activation_derivative;

	// 损失函数和导数
    function<double(const vector<double>&, const vector<double>&)> loss_function;
    function<vector<double>(const vector<double>&, const vector<double>&)> loss_derivative;

public:
	NeuralNetwork() = default;
    NeuralNetwork(const int input_size, const int hidden_size, const int output_size, double lr, const string& act_func, const string& loss_func);

    vector<double> forward(const vector<double>& input, vector<double>& hidden_layer_output);
    void backward(const vector<double>& input, const vector<double>& hidden_layer_output, const vector<double>& output, const vector<double>& target);
    void backward_update_gradients(
        const vector<double>& input, const vector<double>& hidden_layer_output,
        const vector<double>& output, const vector<double>& target,
        vector<vector<double>>& weights_hidden_grad, vector<double>& bias_hidden_grad,
        vector<vector<double>>& weights_output_grad, vector<double>& bias_output_grad);
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs);
    void train_batch(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs);
    vector<vector<double>> predict(const vector<vector<double>>& input);

    // 矩阵 * 向量
    static vector<double> matmul(const vector<vector<double>>& matrix, const vector<double>& vec);

	// 辅助函数，新建随机矩阵
    static vector<vector<double>> initialize_random_matrix(int rows, int cols, double min_val, double max_val);
};

// 辅助函数
vector<double> relu(const vector<double>& x);
vector<double> relu_derivative(const vector<double>& x);
vector<double> softmax(const vector<double>& x);
vector<double> softmax_derivative(const vector<double>& x);
double cross_entropy(const vector<double>& predicted, const vector<double>& target);
vector<double> cross_entropy_derivative(const vector<double>& predicted, const vector<double>& target);
double mean_squared_error(const vector<double>& predicted, const vector<double>& target);
vector<double> mse_derivative(const vector<double>& predicted, const vector<double>& target);

#endif // NEURAL_NETWORK_H
