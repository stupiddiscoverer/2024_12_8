import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 参数设置
c = 1  # 光速，单位：像素/秒
v = 0.6 * c  # 飞船速度 v/c
mirror_distance = 1  # 镜子距离
time_max = 10  # 动画持续时间，单位：秒

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal', 'box')

# 初始化图形元素
line_light_vertical, = ax.plot([], [], 'b-', lw=2)  # 垂直光钟的光路径
line_light_horizontal, = ax.plot([], [], 'r-', lw=2)  # 水平光钟的光路径
point_origin, = ax.plot([], [], 'ko', markersize=8)  # 光钟原点
line_mirror_vertical, = ax.plot([], [], 'k--', lw=1)  # 垂直光钟镜子
line_mirror_horizontal, = ax.plot([], [], 'k--', lw=1)  # 水平光钟镜子
text_time = ax.text(0, 1.5, '', fontsize=12, ha='center')  # 时间显示

# 动画更新函数
def update(frame):
    # 计算飞船原点位置
    origin_x = v * frame
    origin_y = 0

    # 垂直光钟：光束上下反射
    light_vertical_dist = c * frame
    top_y = light_vertical_dist / 2
    bottom_y = -top_y

    # 水平光钟：光束左右反射
    t1 = mirror_distance / (c - v)
    t2 = mirror_distance / (c + v)
    t_round = t1 + t2
    t_cycle = frame % t_round
    if t_cycle < t1:
        light_horizontal_x = origin_x + c * t_cycle
    else:
        light_horizontal_x = origin_x + mirror_distance - c * (t_cycle - t1)

    # 更新垂直光钟路径
    line_light_vertical.set_data([origin_x, origin_x], [origin_y, top_y])
    line_mirror_vertical.set_data([origin_x, origin_x], [origin_y - mirror_distance, origin_y + mirror_distance])

    # 更新水平光钟路径
    line_light_horizontal.set_data([origin_x, light_horizontal_x], [origin_y, origin_y])

    # 更新时间显示
    text_time.set_text(f'Time = {frame:.2f} s')

    return line_light_vertical, line_light_horizontal, line_mirror_vertical, line_mirror_horizontal, point_origin, text_time

# 创建动画
ani = FuncAnimation(fig, update, frames=np.linspace(0, time_max, 100), interval=50, blit=True)

# 添加原点
point_origin.set_data([0], [0])

# 显示图形
plt.show()
