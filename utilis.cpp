#include "src.h"
using namespace std;

// ε-贪心策略选择动作
int epsilon_greedy(const vector<vector<double>> policy, int state) {
	// policy[state] 为当前状态state下的所有动作值的概率，根据概率选择一个动作
	random_device rd;
	mt19937 gen(rd());  // 随机数种子
	discrete_distribution<> dist(policy[state].begin(), policy[state].end());  // 按P[state]概率分布
	int action = dist(gen);
	return action;
}

// 输出状态值
void printV(GridWorld grid, vector<double> V) {
	for (int i = 0; i < grid.NUM_STATES; i++) {
		cout << std::left << setw(5) << fixed << setprecision(1) << V[i] << " ";
		if ((i + 1) % grid.GRID_COL == 0) {
			cout << endl;
		}
	}
}

//计算L2范式
double norm(vector<double> a, vector<double> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(sum);
}


//计算RMSE
double RMSE(vector<double> a, vector<double> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(sum / a.size());
}

// 归一化

double normalize(double num, int num_min, int num_max) {
	double r = (num - (static_cast<double>(num_max) + num_min) / 2) / (num_max - num_min) * 2;
	return r;
}

// 编号转坐标
pair<double, double> state_to_position(int state, int cols) {
	double row = state / cols + 1;
	double col = state % cols + 1;
	return make_pair(row, col);
}

