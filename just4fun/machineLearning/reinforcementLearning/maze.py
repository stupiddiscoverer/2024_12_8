import numpy as np

# 定义迷宫环境
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])

print(maze.shape)

# 定义Q-table
q_table = np.zeros((9, 10, 4))

# 定义参数
alpha = 0.2  # 学习率
gamma = 0.8  # 折扣因子
epsilon = 0.1  # 探索率
num_episodes = 1000  # 迭代次数

# 定义动作
actions = ['up', 'down', 'left', 'right']

# Q-learning算法
for episode in range(num_episodes):
    state = (8, 1)  # 初始状态
    done = False  # 是否到达终点

    while not done:
        # 判断是否到达终点
        if state == (0, 8):
            done = True
            break
        # 选择动作
        if np.random.uniform() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(q_table[state])]

        # 执行动作
        if action == 'up':
            next_state = (state[0] - 1, state[1])
        elif action == 'down':
            next_state = (state[0] + 1, state[1])
        elif action == 'left':
            next_state = (state[0], state[1] - 1)
        else:
            next_state = (state[0], state[1] + 1)
        if not 0 <= next_state[0] < 9 or not 0 <= next_state[1] < 10:
            print('wrong ', state, next_state, episode)
            q_table[state][actions.index(action)] = -0.1
            continue

        # 更新Q-table
        q_table[state][actions.index(action)] += alpha * (maze[next_state] + gamma * np.max(q_table[next_state]) -
                                                          q_table[state][actions.index(action)])

        # 更新状态
        state = next_state


# 输出最优路径
state = (8, 1)
path = [state]

while state != (0, 8):
    action = actions[np.argmax(q_table[state])]
    if action == 'up':
        state = (state[0] - 1, state[1])
    elif action == 'down':
        state = (state[0] + 1, state[1])
    elif action == 'left':
        state = (state[0], state[1] - 1)
    else:
        state = (state[0], state[1] + 1)
    path.append(state)

print("最优路径：", path)
