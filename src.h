#pragma once
#include <iostream>
#include <vector>
#include <SFML/Graphics.hpp>
#include <unordered_map>
#include <matplotlibcpp.h>
#include <iomanip>
#include <random>
#include <cmath>
#include <initializer_list>
#include <tuple>


using namespace std;
namespace plt = matplotlibcpp;
using namespace sf;

const double gamma = 0.9;
const double alpha = 1.0e-3;
enum Action { STAY = 0, UP = 1, RIGHT = 2, DOWN = 3, LEFT = 4 }; // 定义动作编号


struct Score {
	int TARGET;  // 目标奖励
	int FORBIDDEN;  // 禁区惩罚
	int BORDER;  // 越界惩罚
	int OTHERSTEP;  // 除边界、禁区、目标外，其他地方的回报
};



class GridWorld {

public:
	int GRID_ROW;  // 网格行数
	int GRID_COL;  // 网格列数
	int NUM_STATES;  // 状态数量
	int NUM_ACTIONS;  // 动作数量
	int TARGET;  // 目标奖励
	int FORBIDDEN;  // 禁区惩罚
	int BORDER;  // 越界惩罚
	int OTHERSTEP;  // 除边界、禁区、目标外，其他地方的回报
	std::vector<int> grid;  // 网格结构

	GridWorld(int row, int col, int num_actions, Score score, std::vector<int> ini_grid) {
		GRID_ROW = row;
		GRID_COL = col;
		NUM_ACTIONS = num_actions;
		NUM_STATES = GRID_ROW * GRID_COL;
		BORDER = score.BORDER;
		TARGET = score.TARGET;
		FORBIDDEN = score.FORBIDDEN;
		OTHERSTEP = score.OTHERSTEP;
		grid = ini_grid;
	}

	//每步动作
	std::pair<int, double> step(int state, int action) {
		int row = state / GRID_COL;
		int col = state % GRID_COL;
		double reward = 0;
		switch (action) {
		case UP:
			if (row > 0) {
				reward = grid[state - GRID_COL];
				return { state - GRID_COL, reward };
			}
			reward = BORDER;  // 越界惩罚
			return { state, reward };
		case DOWN:
			if (row < GRID_ROW - 1) {
				reward = grid[state + GRID_COL];
				return { state + GRID_COL, reward };
			}
			reward = BORDER;
			return { state, reward };
		case LEFT:
			if (col > 0) {
				reward = grid[state - 1];
				return { state - 1,reward };
			}
			reward = BORDER;
			return { state, reward };
		case RIGHT:
			if (col < GRID_COL - 1) {
				reward = grid[state + 1];
				return { state + 1,reward };
			}
			reward = BORDER;
			return { state, reward };
		case STAY:
			reward = grid[state];
			return { state, reward };
		default:
			reward = grid[state];
			return { state, reward };
		}
	}

	//输出网格
	void draw_grid() {
		// 创建一个窗口
		sf::RenderWindow window(sf::VideoMode(800, 600), "automation");
		// 定义表格的大小
		const int ROWS = GRID_ROW;
		const int COLS = GRID_COL;
		const float cellWidth = 60.f;
		const float cellHeight = 60.f;
		// 绘制一个ROWS X COLS大小的网格，按照grid把forbidden, target, otherstep分别标记为不同颜色
		sf::RectangleShape cell(sf::Vector2f(cellWidth, cellHeight));
		for (int i = 0; i < ROWS; ++i) {
			for (int j = 0; j < COLS; ++j) {
				int s = i * COLS + j;
				cell.setPosition(j * cellWidth, i * cellHeight);
				if (grid[s] == TARGET) {
					cell.setFillColor(sf::Color::Blue);
				}
				else if (grid[s] == FORBIDDEN) {
					cell.setFillColor(sf::Color::Yellow);
				}
				else {
					cell.setFillColor(sf::Color::White);
				}
				window.draw(cell);
			}
		}
		//生成网格线
		sf::RectangleShape line(sf::Vector2f(800, 1));
		line.setFillColor(sf::Color::Black);
		for (int i = 0; i <= ROWS; ++i) {
			line.setPosition(0, i * cellHeight);
			window.draw(line);
		}
		for (int i = 0; i <= COLS; ++i) {
			line.setPosition(i * cellWidth, 0);
			line.setSize(sf::Vector2f(1, 600));
			window.draw(line);
		}
		window.display();
		while (window.isOpen()) {
			sf::Event event;
			while (window.pollEvent(event)) {
				if (event.type == sf::Event::Closed) {
					window.close();
				}
			}
		}
	}


	//输出策略
	void draw_policy(const std::vector<std::vector<double>> policy) {
		std::vector<int> policy_grid(NUM_STATES, STAY);
		for (int s = 0; s < policy.size(); ++s) {
			policy_grid[s] = distance(policy[s].begin(), max_element(policy[s].begin(), policy[s].end()));
		}
		for (int i = 0; i < GRID_ROW; ++i) {
			for (int j = 0; j < GRID_COL; ++j) {
				int action = policy_grid[i * GRID_COL + j];
				switch (action) {
				case UP: std::cout << " ↑ "; break;
				case DOWN: std::cout << " ↓ "; break;
				case LEFT: std::cout << " ← "; break;
				case RIGHT: std::cout << " → "; break;
				case STAY: std::cout << " S  "; break;
				}
			}
			std::cout << std::endl;
		}
	}
};


void printV(GridWorld grid, vector<double> V);
double norm(vector<double> a, vector<double> b);
double RMSE(vector<double> a, vector<double> b);
double normalize(double num, int num_min, int num_max);
pair<double, double> state_to_position(int state, int cols);

vector<vector<double>> construct_matrices(GridWorld grid, vector<double>& V);
vector<double> bellman(GridWorld grid, vector<vector<double>> policy);
tuple<vector<vector<double>>, vector<double>, vector<double>> value_iteration(GridWorld grid, vector<vector<double>> policy);
tuple<vector<vector<double>>, vector<double>, vector<double>> policy_iteration(GridWorld grid, vector<vector<double>> policy);
tuple<vector<vector<double>>, vector<double>, vector<double>> truncated_policy_iteration(GridWorld grid, vector<vector<double>> policy, int MAX_ITER);
void draw_vlist(vector<double> V1, vector<double> V2, vector<double> V3);
void draw_error(vector<double> errors, int xlen, int n);
void draw_error(vector<double> errors, int xlen, int n, string ylabel);

int epsilon_greedy(const vector<vector<double>> policy, int state);
vector<vector<double>> mc_epsilon_greedy(GridWorld grid, vector<vector<double>> policy, int num_episodes, int T, double EPSILON);


void drawPoint(GridWorld gridworld, vector<vector<double>> policy, int T);
void drawPoint_million(GridWorld gridworld, vector<vector<double>> policy, int T);
void drawPolicy(GridWorld gridworld, vector<vector<double>> policy, vector<double> V);

vector<vector<double>> qlearning_off(GridWorld grid, vector<vector<double>> policy_b, int num_episodes, int T, double learning_rate);

vector<double> td_linear(GridWorld grid, vector<vector<double>> policy, int num_episodes, int T, double learning_rate, int degree);