#include "src.h"
using namespace std;

// ��-̰�Ĳ���ѡ����
int epsilon_greedy(const vector<vector<double>> policy, int state) {
	// policy[state] Ϊ��ǰ״̬state�µ����ж���ֵ�ĸ��ʣ����ݸ���ѡ��һ������
	random_device rd;
	mt19937 gen(rd());  // ���������
	discrete_distribution<> dist(policy[state].begin(), policy[state].end());  // ��P[state]���ʷֲ�
	int action = dist(gen);
	return action;
}

// ���״ֵ̬
void printV(GridWorld grid, vector<double> V) {
	for (int i = 0; i < grid.NUM_STATES; i++) {
		cout << std::left << setw(5) << fixed << setprecision(1) << V[i] << " ";
		if ((i + 1) % grid.GRID_COL == 0) {
			cout << endl;
		}
	}
}

//����L2��ʽ
double norm(vector<double> a, vector<double> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(sum);
}


//����RMSE
double RMSE(vector<double> a, vector<double> b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(sum / a.size());
}

// ��һ��

double normalize(double num, int num_min, int num_max) {
	double r = (num - (static_cast<double>(num_max) + num_min) / 2) / (num_max - num_min) * 2;
	return r;
}

// ���ת����
pair<double, double> state_to_position(int state, int cols) {
	double row = state / cols + 1;
	double col = state % cols + 1;
	return make_pair(row, col);
}

