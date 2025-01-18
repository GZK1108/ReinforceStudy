#include "src.h"

using namespace std;
namespace plt = matplotlibcpp;

const int ID =17; 




// 计算动作值Q
vector<vector<double>> construct_matrices(GridWorld grid, vector<double>& V) {
	int NUM_STATES = grid.NUM_STATES;
	int NUM_ACTIONS = grid.NUM_ACTIONS;
    vector<vector<double>> Q(NUM_STATES, vector<double>(NUM_ACTIONS, 0));
    for (int s = 0; s < NUM_STATES; ++s) {
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            int next;
            double reward;
			tie(next, reward) = grid.step(s, a);
			Q[s][a] = reward + gamma * V[next];
        }
    }
    return Q;
}


//贝尔曼方程迭代法求解
vector<double> bellman(GridWorld grid, vector<vector<double>> policy) {
    int NUM_STATES = grid.NUM_STATES;
    int NUM_ACTIONS = grid.NUM_ACTIONS;
    vector<double> V(NUM_STATES, 0);
    while (true) {
        vector<double> V_new(NUM_STATES, 0);
        vector<vector<double>> Q = construct_matrices(grid, V);
		for (int s = 0; s < NUM_STATES; s++) {
			for (int a = 0; a < NUM_ACTIONS; a++) {
				V_new[s] += policy[s][a] * Q[s][a];
			}
		}
		
		// 计算V_new和V的L2范式
        double diff = norm(V_new, V);
		if (diff < alpha) {
			break;
		}
        V = V_new;
    }
    return V;
}



// 求解贝尔曼最优方程
// 值迭代

tuple<vector<vector<double>>, vector<double>, vector<double>> value_iteration(GridWorld grid, vector<vector<double>> policy) {
	cout << "Value Iteration" << endl;
    int NUM_STATES = grid.NUM_STATES;
	int NUM_ACTIONS = grid.NUM_ACTIONS;
    vector<double> V(NUM_STATES, 0);
    vector<double> Vlist;  //记录每次迭代的V0值
	Vlist.push_back(V[ID]);

	vector<double> V_new(NUM_STATES, 0);
    while (true) {
        // 计算每个状态-动作对的 Q 
        vector<double> V_new(NUM_STATES, 0);
        vector<vector<double>> Q = construct_matrices(grid, V);
        
        // 更新 V(s) = max_a Q(s, a) 并记录最优动作
        for (int s = 0; s < NUM_STATES; ++s) {
            int best_action = distance(Q[s].begin(), max_element(Q[s].begin(), Q[s].end()));
            V_new[s] = Q[s][best_action];  // 获取最大值及对应的动作列索引  25
			//非最优动作概率为0
			for (int a = 0; a < NUM_ACTIONS; ++a) {
				if (a != best_action) {
					policy[s][a] = 0;
				}
                else {
					policy[s][a] = 1;
                }
			}
        }

        Vlist.push_back(V_new[ID]);

        // 计算V_new和V的L2范式
        double diff = norm(V_new, V);
        if (diff < alpha) {
            break;
        }
        V = V_new;
    }

    // 输出最优 V 值
    cout << "Optimal State Values (V*):" << endl;
	printV(grid,V);

    // 输出最优策略
    cout << "Optimal Policy (Actions):" << endl;
    grid.draw_policy(policy);
    
	return make_tuple(policy, V, Vlist);  //返回最优策略、最优状态值、每次迭代的V0值
}


// 求解贝尔曼最优方程
// 策略迭代
tuple<vector<vector<double>>, vector<double>, vector<double>> policy_iteration(GridWorld grid, vector<vector<double>> policy) {
    cout << "Policy Iteration" << endl;
    int NUM_STATES = grid.NUM_STATES;
    int NUM_ACTIONS = grid.NUM_ACTIONS;
    vector<double> V(NUM_STATES, 0);
    vector<double> Vlist;  //记录每次迭代的V0值
    Vlist.push_back(V[ID]);
    //double mean = accumulate(V.begin(), V.end(), 0.0) / V.size();
    //Vlist.push_back(mean);

    while (true) {
        // 策略评估
        vector<double> V_new = V;
        while(true) {
            vector<double> V_iter(NUM_STATES, 0);
            vector<vector<double>> Q = construct_matrices(grid, V_new);
            for (int s = 0; s < NUM_STATES; s++) {
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    V_iter[s] += policy[s][a] * Q[s][a];
                }
            }
            if (norm(V_new, V_iter) < alpha) {
                break;
            }
            V_new = V_iter;
        }
        // 策略改进
        vector<vector<double>> Q = construct_matrices(grid, V_new);
        for (int s = 0; s < NUM_STATES; ++s) {
            int best_action = distance(Q[s].begin(), max_element(Q[s].begin(), Q[s].end()));
            //非最优动作概率为0
            for (int a = 0; a < NUM_ACTIONS; ++a) {
                if (a != best_action) {
                    policy[s][a] = 0;
                }
                else {
                    policy[s][a] = 1;
                }
            }
        }
        //double temp_mean = accumulate(V_new.begin(), V_new.end(), 0.0) / V_new.size();
        Vlist.push_back(V_new[ID]);
        // 如果||V_k+1 - V_k|| < α 则停止
        if (norm(V_new, V) < alpha) {
            break;
        }
        V = V_new;
    }
    // 输出最优 V 值
    cout << "Optimal State Values (V*):" << endl;
    printV(grid,V);

    // 输出最优策略
    cout << "Optimal Policy (Actions):" << endl;
    grid.draw_policy(policy);

	return make_tuple(policy, V, Vlist);
}




// 求解贝尔曼最优方程
// 截断式迭代
tuple<vector<vector<double>>, vector<double>, vector<double>> truncated_policy_iteration(GridWorld grid, vector<vector<double>> policy, int MAX_ITER) {
	cout << "Truncated Policy Iteration" << endl;
    int NUM_STATES = grid.NUM_STATES;
    int NUM_ACTIONS = grid.NUM_ACTIONS;
    vector<double> V(NUM_STATES, 0);
    vector<double> Vlist;  //记录每次迭代的V0值
    Vlist.push_back(V[ID]);
    //double mean = accumulate(V.begin(), V.end(), 0.0) / V.size();
    //Vlist.push_back(mean);
	
    //*******************************************************
    vector<double> errors;  // 记录每次迭代的误差
    vector<vector<double>> temp_policy;vector<double> Vstar;vector<double> temp_Vlist;
	tie(temp_policy, Vstar, temp_Vlist) = policy_iteration(grid, policy);
    Vlist.push_back(V[ID]);
	errors.push_back(norm(V, Vstar));
    //*******************************************************
    while (true) {
        vector<double> V_new = V;
        // 策略评估
        for (int i = 0; i < MAX_ITER; ++i) {
			vector<double> V_iter(NUM_STATES, 0);
            vector<vector<double>> Q = construct_matrices(grid, V_new);
            for (int s = 0; s < NUM_STATES; s++) {
                for (int a = 0; a < NUM_ACTIONS; a++) {
                    V_iter[s] += policy[s][a] * Q[s][a];
                }
            }
            if (norm(V_new, V_iter) < alpha) {
                break;
            }
            V_new = V_iter;
        }

        // 策略改进
        vector<vector<double>> Q = construct_matrices(grid, V_new);

        for (int s = 0; s < NUM_STATES; ++s) {
            int best_action = distance(Q[s].begin(), max_element(Q[s].begin(), Q[s].end()));
            //非最优动作概率为0
            for (int a = 0; a < NUM_ACTIONS; ++a) {
                if (a != best_action) {
                    policy[s][a] = 0;
                }
				else {
					policy[s][a] = 1;
				}
            }
        }
        //double temp_mean = accumulate(V_new.begin(), V_new.end(), 0.0) / V_new.size();
        Vlist.push_back(V_new[ID]);
		errors.push_back(norm(V_new, Vstar));
        // 计算V_new和V的L2范式
        if (norm(V_new, V) < alpha) {
            break;
        }
        V = V_new;
    }
    // 输出最优 V 值
    cout << "Optimal State Values (V*):" << endl;
    printV(grid,V);

    // 输出最优策略
    cout << "Optimal Policy (Actions):" << endl;
    grid.draw_policy(policy);
	draw_error(errors, 60, MAX_ITER);
    return make_tuple(policy, V, Vlist);
}


//迭代绘图函数
void draw_vlist(vector<double> V1, vector<double> V2, vector<double> V3) {
    //把V1,V2,V3补充为相同长度
    int len1 = V1.size(), len2 = V2.size(), len3 = V3.size();
    int len = max(len1, max(len2, len3));
    for (int i = len1; i < len; i++) {
        V1.push_back(V1[len1 - 1]);
    }
    for (int i = len2; i < len; i++) {
        V2.push_back(V2[len2 - 1]);
    }
    for (int i = len3; i < len; i++) {
        V3.push_back(V3[len3 - 1]);
    }
    //v1,v2,v3取前50个数值
    vector<double> V1_50(V1.begin(), V1.begin() + 50);
    vector<double> V2_50(V2.begin(), V2.begin() + 50);
    vector<double> V3_50(V3.begin(), V3.begin() + 50);
    //绘制折线图，使用蓝色圆点，曲线连接
    plt::plot(V1_50, "bo-");
    plt::plot(V2_50, "ro-");
    plt::plot(V3_50, "go-");
    //plt::title("Value");
    plt::xlabel("Iteration");
    plt::ylabel("Value");
    plt::show();
}


// 截断式策略迭代误差绘制函数
void draw_error(vector<double> errors, int xlen, int n, string ylabel) {
	//绘制折线图，使用蓝色圆点，曲线连接
    plt::xlim(0, xlen);
    //plt::ylim(35, 40);
	plt::plot(errors, {{"color","b"},{"markersize","1.0"}});
    plt::xlabel("Episode index");
	plt::ylabel(ylabel);
	
    //plt::title("Truncated Policy Iteration - " + std::to_string(n));
	//保存到文件
	//plt::save("x" + std::to_string(x) + ".png");
	plt::show();
    
}

void draw_error(vector<double> errors, int xlen, int n) {
    //绘制折线图，使用蓝色圆点，曲线连接
    plt::xlim(0, xlen);
    //plt::ylim(35, 40);
    plt::plot(errors, { {"color","b"},{"markersize","1.0"} });
    plt::xlabel("Episode index");
    plt::ylabel("Error");

    //plt::title("Truncated Policy Iteration - " + std::to_string(n));
    //保存到文件
    //plt::save("x" + std::to_string(x) + ".png");
    plt::show();

}

