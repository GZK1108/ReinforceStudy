#include "nn.h"


// ����������
NeuralNetwork::NeuralNetwork(const int input_size, const int hidden_size, const int output_size, double lr, const string& act_func, const string& loss_func)
    : learning_rate(lr) {

    // ѡ�񼤻��
    if (act_func == "relu") {
        activation_function = relu;
        activation_derivative = relu_derivative;
    }
    else {
        activation_function = softmax;
        activation_derivative = softmax_derivative;
    }

    // ��ʧ����
    if (loss_func == "mean_squared_error") {
		loss_function = mean_squared_error;
		loss_derivative = mse_derivative;
    }
    else {
        loss_function = cross_entropy;
        loss_derivative = cross_entropy_derivative;
    }

	// �����ʼ��Ȩ�غ�ƫ��
	weights_hidden = initialize_random_matrix(hidden_size, input_size, -1, 1);
    bias_hidden = vector<double>(hidden_size, 0.0);
	weights_output = initialize_random_matrix(output_size, hidden_size, -1, 1);
    bias_output = vector<double>(output_size, 0.0);
    // ��ʼ�������ݶ�
    batch_weights_hidden_grad= vector<vector<double>>(weights_hidden.size(), vector<double>(weights_hidden[0].size(), 0.0));
    batch_bias_hidden_grad= vector<double>(bias_hidden.size(), 0.0);
    batch_weights_output_grad= vector<vector<double>>(weights_output.size(), vector<double>(weights_output[0].size(), 0.0));
    batch_bias_output_grad= vector<double>(bias_output.size(), 0.0);
}

// ǰ�򴫲�
vector<double> NeuralNetwork::forward(const vector<double>& input, vector<double>& hidden_layer_output) {
    hidden_layer_output = matmul(weights_hidden, input);
    for (size_t i = 0; i < hidden_layer_output.size(); ++i) {
        hidden_layer_output[i] += bias_hidden[i];
    }
    hidden_layer_output = activation_function(hidden_layer_output);

    vector<double> output_layer = matmul(weights_output, hidden_layer_output);
    for (size_t i = 0; i < output_layer.size(); ++i) {
        output_layer[i] += bias_output[i];
    }
    return output_layer;
}

// ���򴫲�������Ȩ�غ�ƫ��
void NeuralNetwork::backward(const vector<double>& input, const vector<double>& hidden_layer_output, const vector<double>& output, const vector<double>& target) {
	// ������������
    vector<double> output_error = loss_derivative(output, target);

	// ���������Ȩ�غ�ƫ��
    for (size_t i = 0; i < weights_output.size(); ++i) {
        for (size_t j = 0; j < hidden_layer_output.size(); ++j) {
            weights_output[i][j] -= learning_rate * output_error[i] * hidden_layer_output[j];
        }
        bias_output[i] -= learning_rate * output_error[i];
    }

	// �����������
    vector<double> hidden_error(hidden_layer_output.size(), 0.0);
    for (size_t i = 0; i < hidden_layer_output.size(); ++i) {
        for (size_t j = 0; j < output_error.size(); ++j) {
            hidden_error[i] += output_error[j] * weights_output[j][i];
        }
        hidden_error[i] *= activation_derivative(hidden_layer_output)[i];
    }

	// ��������Ȩ�غ�ƫ��
    for (size_t i = 0; i < weights_hidden.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            weights_hidden[i][j] -= learning_rate * hidden_error[i] * input[j];
        }
        bias_hidden[i] -= learning_rate * hidden_error[i];
    }
}

// ���򴫲��������ݶ�
void NeuralNetwork::backward_update_gradients(
    const vector<double>& input, const vector<double>& hidden_layer_output,
    const vector<double>& output, const vector<double>& target,
    vector<vector<double>>& weights_hidden_grad, vector<double>& bias_hidden_grad,
    vector<vector<double>>& weights_output_grad, vector<double>& bias_output_grad)
{
    // ��������
    vector<double> output_error = loss_derivative(output, target);

    // �ۻ�������ݶ�
    for (size_t i = 0; i < weights_output.size(); ++i) {
        for (size_t j = 0; j < hidden_layer_output.size(); ++j) {
            weights_output_grad[i][j] += output_error[i] * hidden_layer_output[j];
        }
        bias_output_grad[i] += output_error[i];
    }

    // �����������
    vector<double> hidden_error(hidden_layer_output.size(), 0.0);
    for (size_t i = 0; i < hidden_layer_output.size(); ++i) {
        for (size_t j = 0; j < output_error.size(); ++j) {
            hidden_error[i] += output_error[j] * weights_output[j][i];
        }
        hidden_error[i] *= activation_derivative(hidden_layer_output)[i];
    }

    // �ۻ������ݶ�
    for (size_t i = 0; i < weights_hidden.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            weights_hidden_grad[i][j] += hidden_error[i] * input[j];
        }
        bias_hidden_grad[i] += hidden_error[i];
    }
}

// �漴�ݶ��½�����ѵ��
void NeuralNetwork::train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
    for (int i = 0; i < epochs; ++i) {
        double epoch_loss = 0.0;
		for (size_t j = 0; j < inputs.size(); ++j) {
			vector<double> hidden_layer_output;
			vector<double> output = forward(inputs[j], hidden_layer_output);
			epoch_loss += loss_function(output, targets[j]);
			backward(inputs[j], hidden_layer_output, output, targets[j]);
		}
		//cout << "Epoch " << i + 1 << ", Loss: " << epoch_loss / inputs.size() << endl;
    }
}


// batchѵ��
void NeuralNetwork::train_batch(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
    //�÷���Ŀǰ�е�����⣬��ֹʹ��
	throw invalid_argument("This method is not implemented yet.");
    size_t total_samples = inputs.size();
    if (total_samples != targets.size()) {
        throw invalid_argument("Inputs and targets size mismatch.");
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        // ����ǰ�򴫲�
        vector<vector<double>> hidden_layer_outputs(total_samples);
        vector<vector<double>> outputs(total_samples);

        for (size_t i = 0; i < total_samples; ++i) {
            outputs[i] = forward(inputs[i], hidden_layer_outputs[i]);
            epoch_loss += loss_function(outputs[i], targets[i]);
        }

        

        // �������򴫲�
        for (size_t i = 0; i < total_samples; ++i) {
            backward_update_gradients(
                inputs[i], hidden_layer_outputs[i], outputs[i], targets[i],
                batch_weights_hidden_grad, batch_bias_hidden_grad,
                batch_weights_output_grad, batch_bias_output_grad
            );
        }

        // ������ƽ���ݶȸ���Ȩ�غ�ƫ��
        double batch_size_inv = 1.0 / total_samples;
        for (size_t j = 0; j < weights_output.size(); ++j) {
            for (size_t k = 0; k < weights_output[0].size(); ++k) {
                weights_output[j][k] -= learning_rate * batch_weights_output_grad[j][k] * batch_size_inv;
            }
            bias_output[j] -= learning_rate * batch_bias_output_grad[j] * batch_size_inv;
        }

        for (size_t j = 0; j < weights_hidden.size(); ++j) {
            for (size_t k = 0; k < weights_hidden[0].size(); ++k) {
                weights_hidden[j][k] -= learning_rate * batch_weights_hidden_grad[j][k] * batch_size_inv;
            }
            bias_hidden[j] -= learning_rate * batch_bias_hidden_grad[j] * batch_size_inv;
        }

        // ��ӡÿ�� epoch ��ƽ����ʧ
        cout << "Epoch " << epoch + 1 << ", Loss: " << epoch_loss / total_samples << endl;
    }
}


// Ԥ��
vector<vector<double>> NeuralNetwork::predict(const vector<vector<double>>& input) {
	vector<vector<double>> result(input.size());
	for (int i = 0; i < input.size(); i++) {
		vector<double> hidden_layer_output;
		result[i] = forward(input[i], hidden_layer_output);
	}
    return result;
}

// ���� * ����
vector<double> NeuralNetwork::matmul(const vector<vector<double>>& matrix, const vector<double>& vec) {
    vector<double> result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

// �½��������
vector<vector<double>> NeuralNetwork::initialize_random_matrix(int rows, int cols, double min_val, double max_val) {
    random_device rd;
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(min_val, max_val);

    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}


// ReLU�����
vector<double> relu(const vector<double>& x) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = max(0.0, x[i]);
    }
    return result;
}

// ReLU��
vector<double> relu_derivative(const vector<double>& x) {
    vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0 ? 1.0 : 0.0;
    }
    return result;
}

// Softmax
vector<double> softmax(const vector<double>& x) {
	vector<double> result(x.size());
	double sum = 0.0;
	for (size_t i = 0; i < x.size(); ++i) {
		result[i] = exp(x[i]);
		sum += result[i];
	}
	for (size_t i = 0; i < x.size(); ++i) {
		result[i] /= sum;
	}
	return result;
}

// Softmax��
vector<double> softmax_derivative(const vector<double>& x) {
	vector<double> result(x.size());
	for (size_t i = 0; i < x.size(); ++i) {
		result[i] = x[i] * (1 - x[i]);
	}
	return result;
}

// Loss function: Cross-entropy
double cross_entropy(const vector<double>& predicted, const vector<double>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss -= target[i] * log(predicted[i] + 1e-9);
    }
    return loss;
}

// Cross-entropy ��
vector<double> cross_entropy_derivative(const vector<double>& predicted, const vector<double>& target) {
    vector<double> result(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        result[i] = predicted[i] - target[i];
    }
    return result;
}

// Loss function: Mean Squared Error
double mean_squared_error(const vector<double>& predicted, const vector<double>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss += pow(predicted[i] - target[i], 2);
    }
    return loss / predicted.size();
}

// MSE
vector<double> mse_derivative(const vector<double>& predicted, const vector<double>& target) {
    vector<double> result(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        // y(1-y)e
        result[i] = predicted[i] - target[i];
    }
    return result;
}