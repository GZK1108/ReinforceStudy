#include "src.h"
#include "nn.h"
using namespace std;

// 一条记录
struct Transition {
	int state;
	int action;
	double reward;
	int next_state;
};


// 经验回放类
class ReplayMemory {
private:
	int capacity;  // 经验回放容量
	vector<Transition> transitions;  // 经验回放池
	int push_count = 0;  // 记录经验回放次数

public:
	ReplayMemory() = default;
	ReplayMemory(int capacity) : capacity(capacity) {
		transitions.reserve(capacity);
	}

	// 插入
	void push(Transition transition) {
		if (transitions.size() < capacity) {
			transitions.push_back(transition);
		}
		else {
			transitions[push_count % capacity] = transition;
			push_count++;
		}
	}

	// 随机取样
	Transition sample() {
		return transitions[rand() % transitions.size()];
	}

	// 经验回放池大小
	int size() {
		return transitions.size();
	}
};


class DQNAgent {
private:
	GridWorld grid; // 网格世界
	int NUM_STATES;  // 状态数量
	int NUM_ACTIONS;  // 动作数量
	double gamma;  // 折扣因子
	double learning_rate;  // 学习率
	int batch_size;  // 批处理大小
	ReplayMemory memory;  // 经验回放池
	NeuralNetwork policy_model;  // 主网络/策略网络
	NeuralNetwork target_model;  // 目标网络
	int update_count = 0;  // 更新次数
	
public:
	vector<double> TD_errors;  // 记录损失函数值
	DQNAgent() = default;
	DQNAgent(GridWorld grid, double gamma, double learning_rate, int batch_size, int capacity)
		: grid(grid), gamma(gamma), learning_rate(learning_rate), batch_size(batch_size), memory(capacity) {
		NUM_STATES = grid.NUM_STATES;
		NUM_ACTIONS = grid.NUM_ACTIONS;
		// 初始化神经网络
		// 输入层3个神经元（x,y,action），隐藏层100个神经元，输出层1个神经元（q_value），激活函数relu，损失函数均方误差
		policy_model = NeuralNetwork(3, 100, 1, learning_rate, "relu", "mean_squared_error");
		target_model = policy_model;
	}

	// 存入经验缓冲区
	void remenber(int state, int action, double reward, int next_state) {
		Transition transition = { state, action, reward, next_state };
		memory.push(transition);
	}

	// 选择动作
	int take_action(int state) {
		double x, y;
		tie(x, y) = state_to_position(state, grid.GRID_COL);
		// 每输入一个state,action对应一个q值，因此需要将state的每个动作均通过神经网络计算一遍
		vector<vector<double>> state_tensor;
		for (int i = 0; i < NUM_ACTIONS; ++i) {
			vector<double> state_tensor_i = { normalize(x,1,5), normalize(y,1,5), normalize(double(i),0,4) };
			state_tensor.push_back(state_tensor_i);
		}
		vector<vector<double>> q_values = policy_model.predict(state_tensor);
		vector<double> q_list;
		for (int i = 0; i < q_values.size(); ++i) {
			q_list.push_back(q_values[i][0]);
		}
		// 取最大q值对应的动作
		return distance(q_list.begin(), max_element(q_list.begin(), q_list.end()));
	}

	// 训练
	void update() {
		
		if (memory.size() < batch_size) {
			return;
		}
		// 从经验回放池中取样
		vector<Transition> transitions;
		for (int i = 0; i < batch_size; ++i) {
			transitions.push_back(memory.sample());
		}


		// 计算目标值
		vector<vector<double>> states;
		// 下一个状态的五个动作，便于取最大动作值（可优化结构）
		vector<vector<double>> next_states_0; 
		vector<vector<double>> next_states_1;
		vector<vector<double>> next_states_2;
		vector<vector<double>> next_states_3;
		vector<vector<double>> next_states_4;
		vector<vector<double>> rewards;
		vector<vector<int>> actions;
		// 取出样本结果，存入对应结构
		for (int i = 0; i < transitions.size();++i) {
			Transition trans = transitions[i];
			double x, y;
			tie(x, y) = state_to_position(trans.state, grid.GRID_COL);
			vector<double> state_tensor = { normalize(x,1,5), normalize(y,1,5), normalize(trans.action,0,4) };
			double next_x, next_y;
			tie(next_x, next_y) = state_to_position(trans.next_state, grid.GRID_COL);
			vector<double> next_state_tensor_0 = { normalize(next_x,1,5), normalize(next_y,1,5), normalize(0,0,4) };
			vector<double> next_state_tensor_1 = { normalize(next_x,1,5), normalize(next_y,1,5), normalize(1,0,4) };
			vector<double> next_state_tensor_2 = { normalize(next_x,1,5), normalize(next_y,1,5), normalize(2,0,4) };
			vector<double> next_state_tensor_3 = { normalize(next_x,1,5), normalize(next_y,1,5), normalize(3,0,4) };
			vector<double> next_state_tensor_4 = { normalize(next_x,1,5), normalize(next_y,1,5), normalize(4,0,4) };
			vector<double> reward_tensor = { trans.reward };
			vector<int> action_tensor = { trans.action };
			states.push_back(state_tensor);
			next_states_0.push_back(next_state_tensor_0);
			next_states_1.push_back(next_state_tensor_1);
			next_states_2.push_back(next_state_tensor_2);
			next_states_3.push_back(next_state_tensor_3);
			next_states_4.push_back(next_state_tensor_4);
			rewards.push_back(reward_tensor);
			actions.push_back(action_tensor);
		}


		// 选择action对应的q值
		vector<double> q_values;
		for (int i = 0; i < batch_size; ++i) {
			q_values.push_back(policy_model.predict(states)[i][0]);
		}

		// 下一个状态的最大q值
		vector<vector<double>> next_q_matrix_0 = target_model.predict(next_states_0);
		vector<vector<double>> next_q_matrix_1 = target_model.predict(next_states_1);
		vector<vector<double>> next_q_matrix_2 = target_model.predict(next_states_2);
		vector<vector<double>> next_q_matrix_3 = target_model.predict(next_states_3);
		vector<vector<double>> next_q_matrix_4 = target_model.predict(next_states_4);
		vector<double> next_q_values;
		for (int i = 0; i < batch_size; ++i) {
			next_q_values.push_back(max({ next_q_matrix_0[i][0], next_q_matrix_1[i][0], next_q_matrix_2[i][0], next_q_matrix_3[i][0], next_q_matrix_4[i][0] }));
		}
		// 计算目标值
		vector<double> target_values;
		for (int i = 0; i < batch_size; ++i) {
			target_values.push_back(rewards[i][0] + gamma * next_q_values[i]);
		}
		// 计算损失函数损失
		double loss_values = mean_squared_error(q_values, target_values);
		if (loss_values != loss_values) {
			throw invalid_argument("Error loss_values");
		}
		cout << "loss: " << loss_values << endl;
		TD_errors.push_back(loss_values);
		// 把target_values转换成二维数组，便于传入神经网络（可优化神经网络设计）
		vector<vector<double>> target_matrix = vector<vector<double>>(batch_size, vector<double>(1, 0));
		for (int i = 0; i < batch_size; ++i) {
			target_matrix[i][0] = target_values[i];
		}
		// 更新神经网络
		policy_model.train(states, target_matrix, 20);

		// 更新target_model
		if (update_count % 20 == 0) {
			target_model = policy_model;
			//cout << "update target model" << endl;
		}
		update_count++;
	}

};


/*
int main() {
	
	const int GRID_ROW = 5;  // 网格行数
	const int GRID_COL = 5;  //网络列数
	const int NUM_STATES = GRID_ROW * GRID_COL;  // 状态数量
	const int NUM_ACTIONS = 5;  // 动作数量
	const int BORDER = -10;  // 越界惩罚
	const int FORBIDDEN = -10;  //禁区惩罚
	const int TARGET = 1;  //目标奖励
	const int OTHERSTEP = 0; //除边界、禁区、目标外，其他地方的回报
	const int T = 100; // 每个episode的步长
	const int NUM_EPISODES = 1;  // 采样次数
	const double learning_rate = 0.0005; // 学习率alpha



	vector<int> grid = {
		OTHERSTEP, OTHERSTEP, OTHERSTEP, OTHERSTEP, OTHERSTEP,
		OTHERSTEP, FORBIDDEN, FORBIDDEN, OTHERSTEP, OTHERSTEP,
		OTHERSTEP, OTHERSTEP, FORBIDDEN, OTHERSTEP, OTHERSTEP,
		OTHERSTEP, FORBIDDEN, TARGET,    FORBIDDEN, OTHERSTEP,
		OTHERSTEP, FORBIDDEN, OTHERSTEP, OTHERSTEP, OTHERSTEP
	};
	Score score = { TARGET, FORBIDDEN, BORDER, OTHERSTEP };
	GridWorld gridworld(GRID_ROW, GRID_COL, NUM_ACTIONS, score, grid);
	vector<vector<double>> policy = vector<vector<double>>(NUM_STATES, vector<double>(NUM_ACTIONS, 0.2));
	vector<double> Vstar, Vlist;
	vector<vector<double>> policy_star;

	tie(policy_star, Vstar, Vlist) = policy_iteration(gridworld, policy);
	vector<double> errors;
	//drawPolicy(gridworld, policy, V);
	DQNAgent agent(gridworld, 0.9, 0.001, 100, 1000);

	for (int i = 0; i < NUM_EPISODES; ++i) {
		int state = 0;
		for (int t = 0; t < T; ++t) {
			int action = epsilon_greedy(policy, state);
			int next_state;
			double reward;
			tie(next_state, reward) = gridworld.step(state, action);
			agent.remenber(state, action, reward, next_state);
			state = next_state;
		}


		for (int i = 0; i < 1000; ++i) {
			cout << "episode: " << i << endl;
			agent.update();
			vector<vector<double>> target_policy(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
			for (int s = 0; s < NUM_STATES; ++s) {
				int action = agent.take_action(s);
				target_policy[s][action] = 1.0;
			}
			vector<double> V = bellman(gridworld, target_policy);
			errors.push_back(RMSE(V, Vstar));
			
		}
		draw_error(errors, 1000, 0, "RMSE");
		draw_error(agent.TD_errors, 1000, 0, "loss function");
	}
	vector<vector<double>> target_policy(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
	for (int s = 0; s < NUM_STATES; ++s) {
		int action = agent.take_action(s);
		target_policy[s][action] = 1.0;
	}
	vector<double> V = bellman(gridworld, target_policy);
	drawPolicy(gridworld, target_policy, V);
	

	return 0;
}
*/