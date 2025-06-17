import numpy as np
import matplotlib.pyplot as plt


def decode_dna(x):
    X_BOUND = [-3, 2]
    return x.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]


# 目标函数
def fitness_func(pop):  # (POP_SIZE, DNA_SIZE)
    x = decode_dna(pop)
    return x * (x + 1) * (x - 1) + 3*2*4  # 必须要正数


# 参数设置
POP_SIZE = 50
DNA_SIZE = 20
CROSS_RATE = 0.8
MUTATION_RATE = 0.01
N_GENERATIONS = 100


# 选择、交叉、变异函数
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, DNA_SIZE).astype(bool)
        parent[cross_points] = pop[i, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] ^= 1
    return child


# 初始化种群
pop = np.random.randint(0, 2, (POP_SIZE, DNA_SIZE))
best_individual = None
best_fitness = -np.inf
best_fitness_history = []

# 迭代优化
for _ in range(N_GENERATIONS):
    fitness = fitness_func(pop)
    current_max_fitness = np.max(fitness)
    current_best_idx = np.argmax(fitness)

    # 更新历史最优解
    if current_max_fitness > best_fitness:
        best_fitness = current_max_fitness
        best_individual = pop[current_best_idx].copy()  # 保存二进制DNA

    best_fitness_history.append(best_fitness)
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

# 可视化
plt.plot(best_fitness_history)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.show()


# 输出结果（完整打印二进制DNA和x值）
current_best_idx = np.argmax(fitness_func(pop))
current_best_dna = pop[current_best_idx]
current_best_x = decode_dna(current_best_dna)
current_fitness = fitness_func(pop)[current_best_idx]

print("\n=== 当前种群最优解 ===")
print("二进制DNA:", current_best_dna)
print("x值:", current_best_x)
print("适应度:", current_fitness)

print("\n=== 历史最优解 ===")
print("二进制DNA:", best_individual)
print("x值:", decode_dna(best_individual))
print("适应度:", best_fitness)