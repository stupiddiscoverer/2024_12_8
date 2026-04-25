import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numba import cuda

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 参数设置
NUM_BIRDS = 1000  # 适当减少数量以保证性能
WIDTH, HEIGHT = 1800, 900
MAX_SPEED = 3.0
PERCEPTION_RADIUS = 40
SEPARATION_WEIGHT = 1.8
ALIGNMENT_WEIGHT = 1.2
COHESION_WEIGHT = 1.0

# 初始化鸟群
np.random.seed(42)
positions = np.random.rand(NUM_BIRDS, 2).astype(np.float32)
positions[:, 0] *= WIDTH
positions[:, 1] *= HEIGHT

angles = np.random.rand(NUM_BIRDS) * 2 * np.pi
velocities = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32) * MAX_SPEED

# 翅膀动画参数
wing_phases = np.random.rand(NUM_BIRDS) * 2 * np.pi
wing_amplitudes = np.random.rand(NUM_BIRDS) * 0.5 + 0.5


# CUDA核函数 - 鸟群行为计算
@cuda.jit
def boids_kernel(positions, velocities, new_velocities, width, height,
                 perception_radius, separation_weight, alignment_weight, cohesion_weight):
    idx = cuda.grid(1)
    if idx >= positions.shape[0]:
        return

    pos_x, pos_y = positions[idx]
    vel_x, vel_y = velocities[idx]

    separation_x, separation_y = 0.0, 0.0
    alignment_x, alignment_y = 0.0, 0.0
    cohesion_x, cohesion_y = 0.0, 0.0
    neighbor_count = 0

    # 检查所有邻居
    for i in range(positions.shape[0]):
        if i == idx:
            continue

        other_x, other_y = positions[i]
        other_vel_x, other_vel_y = velocities[i]

        # 计算距离（考虑环形世界）
        dx = other_x - pos_x
        dy = other_y - pos_y

        if dx > width / 2:
            dx -= width
        elif dx < -width / 2:
            dx += width
        if dy > height / 2:
            dy -= height
        elif dy < -height / 2:
            dy += height

        distance_sq = dx * dx + dy * dy

        if distance_sq < perception_radius * perception_radius:
            distance = math.sqrt(distance_sq)

            # 分离行为
            if distance < perception_radius / 2:
                separation_x -= dx / distance
                separation_y -= dy / distance

            # 对齐行为
            alignment_x += other_vel_x
            alignment_y += other_vel_y

            # 聚合行为
            cohesion_x += other_x
            cohesion_y += other_y

            neighbor_count += 1

    # 应用行为规则
    if neighbor_count > 0:
        alignment_x /= neighbor_count
        alignment_y /= neighbor_count

        cohesion_x = cohesion_x / neighbor_count - pos_x
        cohesion_y = cohesion_y / neighbor_count - pos_y

        # 归一化
        cohesion_length = math.sqrt(cohesion_x * cohesion_x + cohesion_y * cohesion_y)
        if cohesion_length > 0:
            cohesion_x /= cohesion_length
            cohesion_y /= cohesion_length

    # 更新速度
    new_vel_x = vel_x + (separation_x * separation_weight +
                         alignment_x * alignment_weight +
                         cohesion_x * cohesion_weight)
    new_vel_y = vel_y + (separation_y * separation_weight +
                         alignment_y * alignment_weight +
                         cohesion_y * cohesion_weight)

    # 限制速度
    speed = math.sqrt(new_vel_x * new_vel_x + new_vel_y * new_vel_y)
    if speed > MAX_SPEED:
        new_vel_x = (new_vel_x / speed) * MAX_SPEED
        new_vel_y = (new_vel_y / speed) * MAX_SPEED

    new_velocities[idx, 0] = new_vel_x
    new_velocities[idx, 1] = new_vel_y


@cuda.jit
def update_positions_kernel(positions, velocities, width, height):
    idx = cuda.grid(1)
    if idx >= positions.shape[0]:
        return

    positions[idx, 0] += velocities[idx, 0]
    positions[idx, 1] += velocities[idx, 1]

    # 环形边界
    if positions[idx, 0] < 0:
        positions[idx, 0] += width
    elif positions[idx, 0] >= width:
        positions[idx, 0] -= width

    if positions[idx, 1] < 0:
        positions[idx, 1] += height
    elif positions[idx, 1] >= height:
        positions[idx, 1] -= height
    # # 精确边界反弹
    # if positions[idx, 0] < 0:
    #     velocities[idx, 0] = abs(velocities[idx, 0])  # 确保向右
    #     positions[idx, 0] = 0
    # elif positions[idx, 0] >= width:
    #     velocities[idx, 0] = -abs(velocities[idx, 0])  # 确保向左
    #     positions[idx, 0] = width - 1
    #
    # if positions[idx, 1] < 0:
    #     velocities[idx, 1] = abs(velocities[idx, 1])  # 确保向上
    #     positions[idx, 1] = 0
    # elif positions[idx, 1] >= height:
    #     velocities[idx, 1] = -abs(velocities[idx, 1])  # 确保向下
    #     positions[idx, 1] = height - 1


# 将数据复制到GPU
d_positions = cuda.to_device(positions)
d_velocities = cuda.to_device(velocities)
d_new_velocities = cuda.device_array_like(velocities)

# CUDA配置
threads_per_block = 256
blocks_per_grid = (NUM_BIRDS + threads_per_block - 1) // threads_per_block

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_title(f'CUDA Bird Flocking Simulation - {NUM_BIRDS:,} Birds', fontsize=14)
ax.set_aspect('equal')

# 初始化空的线段集合
line_collection = LineCollection([], colors='blue', linewidths=0.8, alpha=0.7)
ax.add_collection(line_collection)


def create_bird_shape(position, velocity, wing_phase, wing_amplitude):
    """为每个鸟创建形状：身体（方向线）和翅膀"""
    x, y = position
    vx, vy = velocity

    # 计算鸟的方向角度
    angle = math.atan2(vy, vx)

    # 身体长度
    body_length = 8.0
    # 翅膀长度
    wing_length = 6.0

    # 身体端点
    end_x = x + math.cos(angle) * body_length
    end_y = y + math.sin(angle) * body_length

    # 翅膀动画 - 基于相位和振幅
    wing_angle = wing_amplitude * math.sin(wing_phase) * 0.8

    # 左翅膀
    left_wing_x = x + math.cos(angle + math.pi / 2 + wing_angle) * wing_length * 0.7
    left_wing_y = y + math.sin(angle + math.pi / 2 + wing_angle) * wing_length * 0.7

    # 右翅膀
    right_wing_x = x + math.cos(angle - math.pi / 2 - wing_angle) * wing_length * 0.7
    right_wing_y = y + math.sin(angle - math.pi / 2 - wing_angle) * wing_length * 0.7

    # 创建三条线段：身体、左翅膀、右翅膀
    body_line = [(x, y), (end_x, end_y)]
    left_wing_line = [(x, y), (left_wing_x, left_wing_y)]
    right_wing_line = [(x, y), (right_wing_x, right_wing_y)]

    return [body_line, left_wing_line, right_wing_line]


def update(frame):
    global d_positions, d_velocities, d_new_velocities, wing_phases

    # 更新翅膀相位
    # wing_phases += 0.3

    # 执行鸟群行为计算
    boids_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, d_new_velocities,
        WIDTH, HEIGHT, PERCEPTION_RADIUS,
        SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT
    )

    # 更新速度
    cuda.synchronize()
    d_velocities.copy_to_device(d_new_velocities)

    # 更新位置
    update_positions_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_velocities, WIDTH, HEIGHT
    )

    # 将数据复制回CPU
    positions_cpu = d_positions.copy_to_host()
    velocities_cpu = d_velocities.copy_to_host()

    # 为每个鸟创建形状线段
    all_segments = []
    for i in range(NUM_BIRDS):
        bird_segments = create_bird_shape(
            positions_cpu[i],
            velocities_cpu[i],
            wing_phases[i],
            wing_amplitudes[i]
        )
        all_segments.extend(bird_segments)

    # 更新线段集合
    line_collection.set_segments(all_segments)

    # 根据速度调整颜色
    speeds = np.linalg.norm(velocities_cpu, axis=1)
    # colors = plt.cm.plasma(speeds / MAX_SPEED)
    colors = 'black'  # 纯黑色
    line_collection.set_color(colors)

    return line_collection,


print("Starting CUDA Bird Flocking Simulation with Wing Animation...")
ani = FuncAnimation(fig, update, frames=1000, interval=20, blit=True)
plt.tight_layout()
plt.show()