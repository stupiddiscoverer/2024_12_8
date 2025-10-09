import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import time

# 参数设置 - 减少数量以适应CPU计算
NUM_BIRDS = 100  # CPU版本减少数量
WIDTH, HEIGHT = 1200, 800
MAX_SPEED = 2.5
PERCEPTION_RADIUS = 50
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
BORDER_WEIGHT = 2.0
BORDER_MARGIN = 80

# 初始化鸟群
np.random.seed(42)
positions = np.random.rand(NUM_BIRDS, 2).astype(np.float32)
positions[:, 0] *= WIDTH
positions[:, 1] *= HEIGHT

angles = np.random.rand(NUM_BIRDS) * 2 * np.pi
velocities = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32) * MAX_SPEED


def boids_cpu(positions, velocities, width, height, perception_radius,
              separation_weight, alignment_weight, cohesion_weight, border_weight, border_margin):
    """CPU版本的鸟群行为计算"""
    new_velocities = np.zeros_like(velocities)

    for i in range(len(positions)):
        pos_x, pos_y = positions[i]
        vel_x, vel_y = velocities[i]

        separation_x, separation_y = 0.0, 0.0
        alignment_x, alignment_y = 0.0, 0.0
        cohesion_x, cohesion_y = 0.0, 0.0
        neighbor_count = 0

        # 检查所有邻居
        for j in range(len(positions)):
            if i == j:
                continue

            other_x, other_y = positions[j]
            other_vel_x, other_vel_y = velocities[j]

            # 计算距离
            dx = other_x - pos_x
            dy = other_y - pos_y

            # 处理环形世界（可选，现在用边界回避）
            # if dx > width / 2: dx -= width
            # elif dx < -width / 2: dx += width
            # if dy > height / 2: dy -= height
            # elif dy < -height / 2: dy += height

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

        # 边界回避行为
        border_x, border_y = 0.0, 0.0

        # 左边界回避
        if pos_x < border_margin:
            border_x += border_weight * (border_margin - pos_x) / border_margin
        # 右边界回避
        elif pos_x > width - border_margin:
            border_x -= border_weight * (pos_x - (width - border_margin)) / border_margin

        # 上边界回避
        if pos_y < border_margin:
            border_y += border_weight * (border_margin - pos_y) / border_margin
        # 下边界回避
        elif pos_y > height - border_margin:
            border_y -= border_weight * (pos_y - (height - border_margin)) / border_margin

        # 更新速度
        new_vel_x = vel_x + (separation_x * separation_weight +
                             alignment_x * alignment_weight +
                             cohesion_x * cohesion_weight +
                             border_x)
        new_vel_y = vel_y + (separation_y * separation_weight +
                             alignment_y * alignment_weight +
                             cohesion_y * cohesion_weight +
                             border_y)

        # 限制速度
        speed = math.sqrt(new_vel_x * new_vel_x + new_vel_y * new_vel_y)
        if speed > MAX_SPEED:
            new_vel_x = (new_vel_x / speed) * MAX_SPEED
            new_vel_y = (new_vel_y / speed) * MAX_SPEED

        new_velocities[i, 0] = new_vel_x
        new_velocities[i, 1] = new_vel_y

    return new_velocities


def update_positions_cpu(positions, velocities, width, height):
    """更新位置并处理边界"""
    positions[:, 0] += velocities[:, 0]
    positions[:, 1] += velocities[:, 1]

    # 确保不会飞出边界
    border_buffer = 2.0
    positions[:, 0] = np.clip(positions[:, 0], border_buffer, width - border_buffer - 1)
    positions[:, 1] = np.clip(positions[:, 1], border_buffer, height - border_buffer - 1)


# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_title(f'CPU Bird Flocking - {NUM_BIRDS:,} Birds', fontsize=14)
ax.set_aspect('equal')

# 使用散点图
scat = ax.scatter(positions[:, 0], positions[:, 1], s=20, c='black', alpha=0.7)

frame_count = 0
start_time = time.time()


def update(frame):
    global positions, velocities, frame_count

    # CPU计算鸟群行为
    new_velocities = boids_cpu(positions, velocities, WIDTH, HEIGHT, PERCEPTION_RADIUS,
                               SEPARATION_WEIGHT, ALIGNMENT_WEIGHT, COHESION_WEIGHT,
                               BORDER_WEIGHT, BORDER_MARGIN)

    # 更新速度
    velocities = new_velocities

    # 更新位置
    update_positions_cpu(positions, velocities, WIDTH, HEIGHT)

    # 更新图形
    scat.set_offsets(positions)

    # 性能统计
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        ax.set_title(f'CPU Bird Flocking - {NUM_BIRDS:,} Birds - FPS: {fps:.1f}')

    return scat,


print("Starting CPU bird flocking simulation...")
print(f"Warning: CPU version may be slow with {NUM_BIRDS} birds")
ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=True)  # 降低帧率
plt.tight_layout()
plt.show()