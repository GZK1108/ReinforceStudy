# 3D绘图，用于强化学习作业
import numpy as np
import matplotlib.pyplot as plt


def draw_3d(v: np.ndarray, title: str, path: str = None):
    # 数据大小：5x5
    rows, cols = 5, 5

    # 创建网格坐标
    x = np.arange(1, cols + 1)  # x 轴：列号
    y = np.arange(1, rows + 1)  # y 轴：行号
    x, y = np.meshgrid(x, y)  # 创建网格

    z = v.reshape(rows, cols)  # 5x5 数据

    # 创建 3D 图形对象
    fig = plt.figure(figsize=(8, 6))  # 设置图形大小
    ax = fig.add_subplot(111, projection='3d')  # 3D 子图

    # 绘制网格图
    ax.plot_surface(
        y, x, z,  # 网格坐标和数据
        cmap='viridis',  # 颜色映射（'viridis' 是渐变配色方案）
        edgecolor='k',  # 网格线颜色
        linewidth=0.5  # 网格线宽度
    )

    # 设置x轴和y轴的刻度
    ax.set_xticks(np.arange(1, cols + 1, 1))
    ax.set_yticks(np.arange(1, rows + 1, 1))
    ax.set_zticks(np.arange(-5, -2, 1))

    # 设置轴标签和标题
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("row")
    ax.set_ylabel("col")

    if path is not None:
        plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    # 把C++代码中生成的V放入draw_3d函数中
    v = np.array([-3.8,
                  -3.9,
                  -3.6,
                  -3.3,
                  -3.1,
                  -3.7,
                  -3.8,
                  -3.5,
                  -3.2,
                  -3.1,
                  -3.7,
                  -3.7,
                  -3.4,
                  -3.1,
                  -3.1,
                  -3.9,
                  -3.7,
                  -3.3,
                  -3.0,
                  -3.1,
                  -4.4,
                  -4.1,
                  -3.6,
                  -3.3,
                  -3.4])
    draw_3d(v, "TD Linear")
