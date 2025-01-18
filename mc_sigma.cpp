#include "src.h"


using namespace std;





// 蒙特卡洛 ε-贪心策略
vector<vector<double>> mc_epsilon_greedy(GridWorld grid, vector<vector<double>> policy, int num_episodes, int T, double EPSILON) {
	int NUM_STATES = grid.NUM_STATES;
	int NUM_ACTIONS = grid.NUM_ACTIONS;

	vector<vector<double>> Q(NUM_STATES, vector<double>(NUM_ACTIONS, -1.0 * T));
    vector<vector<double>> R(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
    vector<vector<int>> R_count(NUM_STATES, vector<int>(NUM_ACTIONS, 0));
    vector<double> V(NUM_STATES, 0);


    for (int n = 0; n < num_episodes; ++n) {
        cout << "episode:" << n << endl;
        // 输出策略
        vector<tuple<int, int, double>> episode;
   

        //随机选择一个状态开始
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(0, NUM_STATES-1);
        int state = dist(gen);

        // 生成一条完整的episode
        for (int i = 0; i < T;++i) {
            int action = epsilon_greedy(policy, state);
            int next_s;
			double reward;
			tie(next_s, reward) = grid.step(state, action);
            //cout << state << " " << action << " " << reward << endl;
            episode.emplace_back(state, action, reward);
            state = next_s;
        }


        // 更新Q(s,a) 值
        double G = 0;
        // 新建变量存储已经访问过的状态-动作对
        for (int t = T - 1; t >= 0; --t) {
            int state, action;
            double reward;
            tie(state, action, reward) = episode[t];
            G = gamma * G + reward;   // 计算回报
            R[state][action] += G;
            R_count[state][action]++;
            Q[state][action] = R[state][action] / R_count[state][action];
        }


        // 更新策略
        for (int s = 0; s < NUM_STATES; ++s) {
            int best_action = distance(Q[s].begin(), max_element(Q[s].begin(), Q[s].end()));
            V[s] = Q[s][best_action];
            for (int a = 0; a < NUM_ACTIONS; ++a) {
                if (a == best_action) {
                    policy[s][a] = 1.0 - EPSILON + EPSILON / NUM_ACTIONS;
                }
                else {
                    policy[s][a] = EPSILON / NUM_ACTIONS;
                }
            }
        }
        grid.draw_policy(policy);
        cout << endl;

    }
	return policy;
}
