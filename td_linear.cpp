#include "src.h"
using namespace std;


// �������������ĵ��
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




// ������������������ʽ������- ʹ��std::vector
vector<double> feature_vector(int state, int degree) {
	double x = state / 5 + 1;
	double y = state % 5 + 1;
    // ��һ��-1 - 1

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
        //ֻ�е�state��Ϊ1������Ϊ0
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
    // ��ʼ������������Ȩ��
    vector<double> w(degree, 0.0);  // ����ģ�Ͳ���
    vector<double> V(NUM_STATES, 0.0);  // ��ʼ״ֵ̬

    //*******************************************************
    vector<double> errors;  // ��¼ÿ�ε��������
    vector<double> Vstar;
    Vstar = bellman(grid, policy);
    errors.push_back(RMSE(V, Vstar));
    //*******************************************************

    // �����������
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, NUM_STATES - 1);

    // ѵ��
    for (int episode = 0; episode < num_episodes; ++episode) {
        int state = dis(gen);
        // ÿ��episode��ִ��T��
        for (int step = 0; step < T; ++step) {
            vector<double> phi_s = feature_vector(state, degree);  // ��ǰ״̬����������
            int action = epsilon_greedy(policy, state);
            double reward;
            int next_s;
            tie(next_s, reward) = grid.step(state, action);
            vector<double> phi_s_next = feature_vector(next_s, degree);  // ��һ״̬����������

            // TD����
            double delta = reward + gamma * dot_product(phi_s_next, w) - dot_product(phi_s, w);
            for (int i = 0; i < degree; ++i) {
                w[i] += learning_rate * delta * phi_s[i];  
            }

            // ���µ�ǰ״̬Ϊ��һ״̬
            state = next_s;
        }
        // ����ƽ���״ֵ̬������ʵ״ֵ̬�Ƚϣ�
        for (int state = 0; state < NUM_STATES; ++state) {
            vector<double> phi_s = feature_vector(state, degree);
            V[state] = dot_product(phi_s, w);
        }
		errors.push_back(RMSE(V, Vstar));
    }
    draw_error(errors, num_episodes, 0);

    return V;
}