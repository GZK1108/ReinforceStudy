
#include "src.h"
#include "nn.h"
using namespace std;


const int GRID_ROW = 5;  // 网格行数
const int GRID_COL = 5;  //网络列数
const int NUM_STATES = GRID_ROW * GRID_COL;  // 状态数量
const int NUM_ACTIONS = 5;  // 动作数量
const int BORDER = -1;  // 越界惩罚
const int FORBIDDEN = -1;  //禁区惩罚
const int TARGET = 1;  //目标奖励
const int OTHERSTEP = 0; //除边界、禁区、目标外，其他地方的回报
const int MAX_ITER = 5; // 最大迭代次数
const int T = 500; // 每个episode的步长
const int NUM_EPISODES = 500;  // 采样次数
const double EPSILON = 1;  // ε-贪心策略选择概率
const double learning_rate = 0.0005; // 学习率alpha



vector<int> grid = {
    OTHERSTEP, OTHERSTEP, OTHERSTEP, OTHERSTEP, OTHERSTEP,
    OTHERSTEP, FORBIDDEN, FORBIDDEN, OTHERSTEP, OTHERSTEP,
    OTHERSTEP, OTHERSTEP, FORBIDDEN, OTHERSTEP, OTHERSTEP,
    OTHERSTEP, FORBIDDEN, TARGET,    FORBIDDEN, OTHERSTEP,
    OTHERSTEP, FORBIDDEN, OTHERSTEP, OTHERSTEP, OTHERSTEP
};



vector<vector<double>> fix2policy(const vector<int> fixedpolicy) {
    vector<vector<double>> policy(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
    for (int s = 0; s < NUM_STATES; ++s) {
        int a = fixedpolicy[s];
        for (int i = 0; i < NUM_ACTIONS; ++i) {
            if (i == a) {
                policy[s][i] = 1.0 - EPSILON + EPSILON / NUM_ACTIONS;
            }
            else {
                policy[s][i] = EPSILON / NUM_ACTIONS;
            }
        }
    }
    return policy;
}

vector<int> fixedPolicy = {
        RIGHT, RIGHT, RIGHT, RIGHT, DOWN,
        UP, UP, RIGHT, RIGHT, DOWN,
        UP, LEFT, DOWN, RIGHT, DOWN,
        UP, RIGHT, STAY, LEFT, DOWN,
        UP, RIGHT, UP, LEFT, LEFT
};

vector<int> fixedPolicy1 = {
        STAY,LEFT , RIGHT, RIGHT, DOWN,
        UP, UP, RIGHT, RIGHT, STAY,
        STAY, LEFT, RIGHT, RIGHT, UP,
        UP, RIGHT, STAY, RIGHT, UP,
        UP, RIGHT, RIGHT, RIGHT, STAY
};

vector<int> fixedPolicy2 = {
        UP,STAY , LEFT, RIGHT, RIGHT,
        RIGHT, RIGHT, LEFT, UP, LEFT,
        RIGHT, LEFT, LEFT, RIGHT, DOWN,
        STAY, LEFT, LEFT, STAY, DOWN,
        LEFT, UP, UP, LEFT, DOWN
};

vector<int> fixedPolicy3(NUM_STATES, RIGHT);


int main() {
    Score score = { TARGET, FORBIDDEN, BORDER, OTHERSTEP };
    GridWorld gridworld(GRID_ROW, GRID_COL, NUM_ACTIONS, score, grid);
	vector<vector<double>> policy = fix2policy(fixedPolicy3);

	//**********************Bellman*******************************
    //vector<double> V1,V2,V3, Vlist1, Vlist2, Vlist3;
	//vector<vector<double>> policy_star_1, policy_star_2, policy_star_3;
	//tie(policy_star_1, V1, Vlist1) = value_iteration(gridworld, policy);
    //drawPolicy(gridworld, policy_star_1, V1);
	//tie(policy_star_2, V2, Vlist2) = policy_iteration(gridworld, policy);
	//drawPolicy(gridworld, policy_star_2, V2);
	//tie(policy_star_3, V3, Vlist3) = truncated_policy_iteration(gridworld, policy);
	//drawPolicy(gridworld, policy_star_3, V3);
	//draw_vlist(Vlist1, Vlist2, Vlist3);


	//**********************MC epsilon*******************************
	//drawPoint(gridworld, policy, T);  // picture
	//drawPoint_million(gridworld, policy, T);  // picture
	//vector<vector<double>> Policy_mc = mc_epsilon_greedy(gridworld, policy, NUM_EPISODES, T, EPSILON);
	//vector<double> V = bellman(gridworld, Policy_mc);
	//drawPolicy(gridworld, Policy_mc, V);


	//**********************Q-Learning*******************************
    //drawPoint(gridworld, policy, T);  // picture
	//vector<vector<double>> Policy_q = qlearning_off(gridworld, policy, NUM_EPISODES, T, learning_rate);
	//vector<double> V_q = bellman(gridworld, Policy_q);
	//drawPolicy(gridworld, Policy_q, V_q);


    //**********************TD-Linear*******************************
    //vector<double> V_td = td_linear(gridworld, policy, NUM_EPISODES, T, learning_rate, 10);
    //drawPolicy(gridworld, policy, V_td);



	return 0;
}
