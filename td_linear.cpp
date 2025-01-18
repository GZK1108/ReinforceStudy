#include "src.h"
using namespace std;


// 计算两个向量的点积
double dot_product(const vector<double>& vec1, const vector<double>& vec2) {
    double result = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

class Normalization {
public:
    int num_min;
    int num_max;
	Normalization(int num_min, int num_max) : num_min(num_min), num_max(num_max) {}
    double normalize(double num) {
        return (num - (num_max + num_min) / 2) / (num_max - num_min) * 2;
    }
};




// 特征向量函数（多项式特征）- 使用std::vector
vector<double> feature_vector(int state, int degree) {
	double x = state / 5 + 1;
	double y = state % 5 + 1;
    // 归一化-1 - 1

    Normalization norma(1, 5);
	x = norma.normalize(x);
	y = norma.normalize(y);
    vector<double> phi(degree);
    switch (degree) {
    case 3:
        phi = { 1, x, y };  // [1, x, y]
        break;
    case 6:
        phi = { 1, x, y, x * x, y * y, x * y };  // [1, x, y, x2, y2, xy]
        break;
    case 10:
        phi = { 1, x, y, x * x, y * y, x * y, x * x * x, y * y * y, x * x * y, x * y * y };  // [1, x, y, x2, y2, xy, x3, y3, x2y, xy2]
        break;
    case 25:
        //只有第state个为1，其他为0
		phi = vector<double>(degree, 0.0);
		phi[state] = 1.0;
		break;
    default:
        phi = vector<double>(degree, 0.0);
    }
    return phi;
}




vector<double> td_linear(GridWorld grid, vector<vector<double>> policy, int num_episodes, int T, double learning_rate, int degree) {
    int NUM_STATES = grid.NUM_STATES;
    int NUM_ACTIONS = grid.NUM_ACTIONS;
    // 初始化特征向量和权重
    vector<double> w(degree, 0.0);  // 线性模型参数
    vector<double> V(NUM_STATES, 0.0);  // 初始状态值

    //*******************************************************
    vector<double> errors;  // 记录每次迭代的误差
    vector<double> Vstar;
    Vstar = bellman(grid, policy);
    errors.push_back(RMSE(V, Vstar));
    //*******************************************************

    // 随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, NUM_STATES - 1);

    // 训练
    for (int episode = 0; episode < num_episodes; ++episode) {
        int state = dis(gen);
        // 每个episode中执行T步
        for (int step = 0; step < T; ++step) {
            vector<double> phi_s = feature_vector(state, degree);  // 当前状态的特征向量
            int action = epsilon_greedy(policy, state);
            double reward;
            int next_s;
            tie(next_s, reward) = grid.step(state, action);
            vector<double> phi_s_next = feature_vector(next_s, degree);  // 下一状态的特征向量

            // TD更新
            double delta = reward + gamma * dot_product(phi_s_next, w) - dot_product(phi_s, w);
            for (int i = 0; i < degree; ++i) {
                w[i] += learning_rate * delta * phi_s[i];  
            }

            // 更新当前状态为下一状态
            state = next_s;
        }
        // 计算逼近的状态值误差（与真实状态值比较）
        for (int state = 0; state < NUM_STATES; ++state) {
            vector<double> phi_s = feature_vector(state, degree);
            V[state] = dot_product(phi_s, w);
        }
		errors.push_back(RMSE(V, Vstar));
    }
    draw_error(errors, num_episodes, 0);

    return V;
}