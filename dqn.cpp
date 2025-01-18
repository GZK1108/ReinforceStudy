#include "src.h"
#include "nn.h"
using namespace std;

// һ����¼
struct Transition {
	int state;
	int action;
	double reward;
	int next_state;
};


// ����ط���
class ReplayMemory {
private:
	int capacity;  // ����ط�����
	vector<Transition> transitions;  // ����طų�
	int push_count = 0;  // ��¼����طŴ���

public:
	ReplayMemory() = default;
	ReplayMemory(int capacity) : capacity(capacity) {
		transitions.reserve(capacity);
	}

	// ����
	void push(Transition transition) {
		if (transitions.size() < capacity) {
			transitions.push_back(transition);
		}
		else {
			transitions[push_count % capacity] = transition;
			push_count++;
		}
	}

	// ���ȡ��
	Transition sample() {
		return transitions[rand() % transitions.size()];
	}

	// ����طųش�С
	int size() {
		return transitions.size();
	}
};


class DQNAgent {
private:
	GridWorld grid; // ��������
	int NUM_STATES;  // ״̬����
	int NUM_ACTIONS;  // ��������
	double gamma;  // �ۿ�����
	double learning_rate;  // ѧϰ��
	int batch_size;  // �������С
	ReplayMemory memory;  // ����طų�
	NeuralNetwork policy_model;  // ������/��������
	NeuralNetwork target_model;  // Ŀ������
	int update_count = 0;  // ���´���
	
public:
	vector<double> TD_errors;  // ��¼��ʧ����ֵ
	DQNAgent() = default;
	DQNAgent(GridWorld grid, double gamma, double learning_rate, int batch_size, int capacity)
		: grid(grid), gamma(gamma), learning_rate(learning_rate), batch_size(batch_size), memory(capacity) {
		NUM_STATES = grid.NUM_STATES;
		NUM_ACTIONS = grid.NUM_ACTIONS;
		// ��ʼ��������
		// �����3����Ԫ��x,y,action�������ز�100����Ԫ�������1����Ԫ��q_value���������relu����ʧ�����������
		policy_model = NeuralNetwork(3, 100, 1, learning_rate, "relu", "mean_squared_error");
		target_model = policy_model;
	}

	// ���뾭�黺����
	void remenber(int state, int action, double reward, int next_state) {
		Transition transition = { state, action, reward, next_state };
		memory.push(transition);
	}

	// ѡ����
	int take_action(int state) {
		double x, y;
		tie(x, y) = state_to_position(state, grid.GRID_COL);
		// ÿ����һ��state,action��Ӧһ��qֵ�������Ҫ��state��ÿ��������ͨ�����������һ��
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
		// ȡ���qֵ��Ӧ�Ķ���
		return distance(q_list.begin(), max_element(q_list.begin(), q_list.end()));
	}

	// ѵ��
	void update() {
		
		if (memory.size() < batch_size) {
			return;
		}
		// �Ӿ���طų���ȡ��
		vector<Transition> transitions;
		for (int i = 0; i < batch_size; ++i) {
			transitions.push_back(memory.sample());
		}


		// ����Ŀ��ֵ
		vector<vector<double>> states;
		// ��һ��״̬���������������ȡ�����ֵ�����Ż��ṹ��
		vector<vector<double>> next_states_0; 
		vector<vector<double>> next_states_1;
		vector<vector<double>> next_states_2;
		vector<vector<double>> next_states_3;
		vector<vector<double>> next_states_4;
		vector<vector<double>> rewards;
		vector<vector<int>> actions;
		// ȡ����������������Ӧ�ṹ
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


		// ѡ��action��Ӧ��qֵ
		vector<double> q_values;
		for (int i = 0; i < batch_size; ++i) {
			q_values.push_back(policy_model.predict(states)[i][0]);
		}

		// ��һ��״̬�����qֵ
		vector<vector<double>> next_q_matrix_0 = target_model.predict(next_states_0);
		vector<vector<double>> next_q_matrix_1 = target_model.predict(next_states_1);
		vector<vector<double>> next_q_matrix_2 = target_model.predict(next_states_2);
		vector<vector<double>> next_q_matrix_3 = target_model.predict(next_states_3);
		vector<vector<double>> next_q_matrix_4 = target_model.predict(next_states_4);
		vector<double> next_q_values;
		for (int i = 0; i < batch_size; ++i) {
			next_q_values.push_back(max({ next_q_matrix_0[i][0], next_q_matrix_1[i][0], next_q_matrix_2[i][0], next_q_matrix_3[i][0], next_q_matrix_4[i][0] }));
		}
		// ����Ŀ��ֵ
		vector<double> target_values;
		for (int i = 0; i < batch_size; ++i) {
			target_values.push_back(rewards[i][0] + gamma * next_q_values[i]);
		}
		// ������ʧ������ʧ
		double loss_values = mean_squared_error(q_values, target_values);
		if (loss_values != loss_values) {
			throw invalid_argument("Error loss_values");
		}
		cout << "loss: " << loss_values << endl;
		TD_errors.push_back(loss_values);
		// ��target_valuesת���ɶ�ά���飬���ڴ��������磨���Ż���������ƣ�
		vector<vector<double>> target_matrix = vector<vector<double>>(batch_size, vector<double>(1, 0));
		for (int i = 0; i < batch_size; ++i) {
			target_matrix[i][0] = target_values[i];
		}
		// ����������
		policy_model.train(states, target_matrix, 20);

		// ����target_model
		if (update_count % 20 == 0) {
			target_model = policy_model;
			//cout << "update target model" << endl;
		}
		update_count++;
	}

};


/*
int main() {
	
	const int GRID_ROW = 5;  // ��������
	const int GRID_COL = 5;  //��������
	const int NUM_STATES = GRID_ROW * GRID_COL;  // ״̬����
	const int NUM_ACTIONS = 5;  // ��������
	const int BORDER = -10;  // Խ��ͷ�
	const int FORBIDDEN = -10;  //�����ͷ�
	const int TARGET = 1;  //Ŀ�꽱��
	const int OTHERSTEP = 0; //���߽硢������Ŀ���⣬�����ط��Ļر�
	const int T = 100; // ÿ��episode�Ĳ���
	const int NUM_EPISODES = 1;  // ��������
	const double learning_rate = 0.0005; // ѧϰ��alpha



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