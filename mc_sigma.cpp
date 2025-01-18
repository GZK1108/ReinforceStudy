#include "src.h"


using namespace std;





// ���ؿ��� ��-̰�Ĳ���
vector<vector<double>> mc_epsilon_greedy(GridWorld grid, vector<vector<double>> policy, int num_episodes, int T, double EPSILON) {
	int NUM_STATES = grid.NUM_STATES;
	int NUM_ACTIONS = grid.NUM_ACTIONS;

	vector<vector<double>> Q(NUM_STATES, vector<double>(NUM_ACTIONS, -1.0 * T));
    vector<vector<double>> R(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
    vector<vector<int>> R_count(NUM_STATES, vector<int>(NUM_ACTIONS, 0));
    vector<double> V(NUM_STATES, 0);


    for (int n = 0; n < num_episodes; ++n) {
        cout << "episode:" << n << endl;
        // �������
        vector<tuple<int, int, double>> episode;
   

        //���ѡ��һ��״̬��ʼ
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(0, NUM_STATES-1);
        int state = dist(gen);

        // ����һ��������episode
        for (int i = 0; i < T;++i) {
            int action = epsilon_greedy(policy, state);
            int next_s;
			double reward;
			tie(next_s, reward) = grid.step(state, action);
            //cout << state << " " << action << " " << reward << endl;
            episode.emplace_back(state, action, reward);
            state = next_s;
        }


        // ����Q(s,a) ֵ
        double G = 0;
        // �½������洢�Ѿ����ʹ���״̬-������
        for (int t = T - 1; t >= 0; --t) {
            int state, action;
            double reward;
            tie(state, action, reward) = episode[t];
            G = gamma * G + reward;   // ����ر�
            R[state][action] += G;
            R_count[state][action]++;
            Q[state][action] = R[state][action] / R_count[state][action];
        }


        // ���²���
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
