#include "src.h"
#include <sstream>
using namespace std;
using namespace sf;
namespace plt = matplotlibcpp;


//动图绘制函数
void drawGrid(RenderWindow& window, GridWorld gridworld, int cellSize) {
    // 定义表格的大小
    const int ROWS = gridworld.GRID_ROW;
    const int COLS = gridworld.GRID_ROW;
    // 绘制一个ROWS X COLS大小的网格，按照grid把forbidden, target, otherstep分别标记为不同颜色
    sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            int s = i * COLS + j;
            cell.setPosition(j * cellSize, i * cellSize);
            if (gridworld.grid[s] == gridworld.TARGET) {
                cell.setFillColor(sf::Color::Blue);
            }
            else if (gridworld.grid[s] == gridworld.FORBIDDEN) {
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
        line.setPosition(0, i * cellSize);
        window.draw(line);
    }
    for (int i = 0; i <= COLS; ++i) {
        line.setPosition(i * cellSize, 0);
        line.setSize(sf::Vector2f(1, 600));
        window.draw(line);
    }
}


// 绘制路径
void drawPoint(GridWorld gridworld, vector<vector<double>> policy, int T) {
    int gridROW = gridworld.GRID_ROW;
    int gridCOL = gridworld.GRID_COL;
    int cellSize = 100;
    // 创建窗口
    RenderWindow window(VideoMode(gridCOL * cellSize, gridROW * cellSize), "Probability Arrows in Grid");
    window.setFramerateLimit(10);


    // 保存路径点
    std::vector<Vector2f> path;
    int state = 0;
    path.emplace_back((1 - 0.5f) * cellSize, (1 - 0.5f) * cellSize);
    for (int i = 0; i < T; i++) {
        int action = epsilon_greedy(policy, state);
        int next_s;
        double reward;
        tie(next_s, reward) = gridworld.step(state, action);

        int x = next_s % gridCOL + 1;  //列
        int y = next_s / gridCOL + 1;  //行
        if (x == 1 && action == LEFT) {
            path.emplace_back((x - 0.8f) * cellSize, (y - 0.5f) * cellSize);
        }
        else if (x == gridCOL && action == RIGHT) {
            path.emplace_back((x - 0.2f) * cellSize, (y - 0.5f) * cellSize);
        }
        else if (y == 1 && action == UP) {
            path.emplace_back((x - 0.5f) * cellSize, (y - 0.8f) * cellSize);
        }
        else if (y == gridROW && action == DOWN) {
            path.emplace_back((x - 0.5f) * cellSize, (y - 0.2f) * cellSize);

        }
        path.emplace_back((x - 0.5f) * cellSize, (y - 0.5f) * cellSize);

        state = next_s;
    }

    // 绘制内容
    window.clear(Color::White);
    // 绘制网格
    drawGrid(window, gridworld, cellSize);

    // 绘制路径
    for (size_t i = 1; i < path.size(); ++i) {
        //随机生成一个0-1之间的数
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0, 1);
        float offset = dis(gen) * 10;
        Vertex thickLine[] = {
            Vertex(Vector2f(path[i - 1].x + offset, path[i - 1].y + offset), Color::Red),
            Vertex(Vector2f(path[i].x + offset, path[i].y + offset), Color::Red)
        };
        window.draw(thickLine, 2, Lines);

    }

    // 显示内容
    window.display();
    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }
    }
}


//绘制路径中每个状态动作对个数
void drawPoint_million(GridWorld gridworld, vector<vector<double>> policy, int T) {
    vector<vector<int>> path(gridworld.NUM_STATES, vector<int>(gridworld.NUM_ACTIONS, 0));
    //固定状态0开始
    int state = 0;
    // 生成一条完整的episode

    for (int i = 0; i < T; ++i) {
        int action = epsilon_greedy(policy, state);
        path[state][action]++;
        int next_s;
        double reward;
        tie(next_s, reward) = gridworld.step(state, action);
        state = next_s;
    }
    //绘图散点图
    vector<double> x, y;
    for (int i = 0; i < gridworld.NUM_STATES; i++) {
        for (int j = 0; j < gridworld.NUM_ACTIONS; j++) {
            x.push_back(i * gridworld.NUM_ACTIONS + j);
            y.push_back(path[i][j]);
        }
    }
    plt::scatter(x, y, 5.0);
    plt::show();
}


//绘制策略
//绘制箭头
void drawArrow(RenderWindow& window, Vector2f start, Vector2f direction, float length, Color color) {
    // 计算箭头的终点
    Vector2f end = start + direction * length;

    // 主线段
    Vertex line[] = {
        Vertex(start, color),
        Vertex(end, color)
    };
    window.draw(line, 2, Lines);

    // 箭头两侧的小线段
    float arrowSize = 5.0f; // 箭头三角形的大小
    Vector2f left = end - direction * arrowSize + Vector2f(-direction.y, direction.x) * (arrowSize / 2.0f);
    Vector2f right = end - direction * arrowSize - Vector2f(-direction.y, direction.x) * (arrowSize / 2.0f);

    Vertex arrowHead[] = {
        Vertex(end, color),
        Vertex(left, color),
        Vertex(end, color),
        Vertex(right, color)
    };
    window.draw(arrowHead, 4, Lines);
}


// 绘制圆圈（表示保持不动的概率）
void drawCircle(RenderWindow& window, Vector2f center, float radius, Color color) {
    CircleShape circle(radius);
    circle.setOrigin(radius, radius); // 设置圆心为绘制中心
    circle.setPosition(center);
    circle.setFillColor(Color::Transparent); // 圆圈内部透明
    circle.setOutlineColor(color);
    circle.setOutlineThickness(2); // 圆圈边框厚度
    window.draw(circle);
}


//绘制策略
void drawPolicy(GridWorld gridworld, vector<vector<double>> policy, vector<double> V) {
    int gridROW = gridworld.GRID_ROW;
    int gridCOL = gridworld.GRID_COL;
    int cellSize = 100;
    // 创建窗口
    RenderWindow window(VideoMode(gridCOL * cellSize, gridROW * cellSize), "Policy");
    window.setFramerateLimit(10);

    //方向
	unordered_map<int, Vector2f> directions = {
		{UP, Vector2f(0, -1)},
		{DOWN, Vector2f(0, 1)},
		{LEFT, Vector2f(-1, 0)},
		{RIGHT, Vector2f(1, 0)},
		{STAY, Vector2f(0, 0)}
	};

    // 主循环
    while (window.isOpen()) {
        Event event;
        while (window.pollEvent(event)) {
            if (event.type == Event::Closed)
                window.close();
        }

        window.clear(Color::White);

        // 绘制网格
        drawGrid(window, gridworld, cellSize);

        // 遍历每个格子，绘制箭头
        for (int s = 0; s < gridworld.NUM_STATES; ++s) {
            //定位
			int row = s / gridCOL;
			int col = s % gridCOL;
            // 格子中心点
            Vector2f center((col + 0.5f) * cellSize, (row + 0.5f) * cellSize);
                
            // 动作：绘制每个方向的箭头，根据概率设置长度
            for (int a = 0; a < gridworld.NUM_ACTIONS; ++a) {
                if (a == STAY) {
                    // 绘制圆圈（保持不动）
                    if (policy[s][a] > 0) {
                        float radius = 0.09 * std::pow(5, policy[s][a]) * cellSize * 0.5; // 圆圈半径与概率成比例
                        drawCircle(window, center, radius, Color::Red);
                    }
                }
                else {
                    if (policy[s][a] > 0) {
                        drawArrow(window, center, directions[a], 0.18 * std::pow(5, policy[s][a]) * cellSize * 0.5, Color::Red);
                    }
                }     
            }   
        }
        window.display();
    }


    //绘制V
    //打开window
    RenderWindow newwindow(VideoMode(gridCOL * cellSize, gridROW * cellSize), "V");
    newwindow.setFramerateLimit(10);
    while (newwindow.isOpen()) {
        Event event;
        while (newwindow.pollEvent(event)) {
            if (event.type == Event::Closed)
                newwindow.close();
        }

        newwindow.clear(Color::White);

        // 绘制网格
        drawGrid(newwindow, gridworld, cellSize);

        // 遍历每个格子，绘制箭头
        for (int s = 0; s < gridworld.NUM_STATES; ++s) {
            //定位
            int row = s / gridCOL;
            int col = s % gridCOL;
			//在每个格子中心绘制V
			Vector2f center((col + 0.5f) * cellSize, (row + 0.5f) * cellSize);
			//绘制V
			Font font;
			font.loadFromFile("C:/Windows/Fonts/arial.ttf");
            
			Text text;
			text.setFont(font);
            // 格式化值为 1 位小数
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(1) << V[s];
            text.setString(stream.str());
			text.setCharacterSize(20);
			text.setFillColor(Color::Black);
			text.setPosition(center.x-10 , center.y-10);
            newwindow.draw(text);
        }
        newwindow.display();
    }
}


