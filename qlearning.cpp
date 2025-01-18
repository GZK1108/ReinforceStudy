#include "src.h"
using namespace std;

// qlearning off-policy
vector<vector<double>> qlearning_off(GridWorld grid, vector<vector<double>> policy_b, int num_episodes, int T, double learning_rate) {
    int NUM_STATES = grid.NUM_STATES;
    int NUM_ACTIONS = grid.NUM_ACTIONS;

    vector<vector<double>> Q(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
    vector<double> V(NUM_STATES, 0);
	vector<vector<double>> policy_t = policy_b;

    //*******************************************************
    vector<double> errors;  // 记录每次迭代的误差
    vector<vector<double>> temp_policy; vector<double> Vstar; vector<double> temp_Vlist;
    tie(temp_policy, Vstar, temp_Vlist) = policy_iteration(grid, policy_b);
    errors.push_back(norm(V, Vstar));
    //*******************************************************

    for (int n = 0; n < num_episodes; ++n) {
        cout << "episode:" << n << endl;
        // 输出策略
        //vector<tuple<int, int, double>> episode;
        int state = 0;
        int length = 0;
        while (length < T) {
            length++;
            int action = epsilon_greedy(policy_b, state);
            int next_s;
            double reward;
            tie(next_s, reward) = grid.step(state, action);

            // 更新Q(s,a) 值
			Q[state][action] = Q[state][action] - learning_rate * (Q[state][action] - (reward + gamma * *max_element(Q[next_s].begin(), Q[next_s].end())));

            // 更新策略
            for (int s = 0; s < NUM_STATES; ++s) {
                int best_action = distance(Q[s].begin(), max_element(Q[s].begin(), Q[s].end()));
				V[s] = Q[s][best_action];
                for (int a = 0; a < NUM_ACTIONS; ++a) {
                    if (a == best_action) {
                        policy_t[s][a] = 1;
                    }
                    else {
                        policy_t[s][a] = 0;
                    }
                }
            }
            state = next_s;
            errors.push_back(norm(V, Vstar));
            
        }
        
        grid.draw_policy(policy_t);
    }
    draw_error(errors, T, 0);
    return policy_t;
}
