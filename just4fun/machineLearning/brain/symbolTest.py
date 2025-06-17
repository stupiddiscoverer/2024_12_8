from gplearn.genetic import SymbolicRegressor
import numpy as np

# 生成数据：y = sin(x) + noise
x = np.random.uniform(-10, 10, 100).reshape(-1, 1)
# y = np.sin(x[:, 0]) + 0.1 * np.random.randn(100)
y = np.sin(x[:, 0])

# 配置符号回归模型
model = SymbolicRegressor(
    population_size=1000,
    generations=20,
    function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
    verbose=1
)

# 训练并输出最佳公式
model.fit(x, y)
print("最佳公式:", model._program)

from pysr import PySRRegressor

model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "exp"],
    model_selection="best"  # 选择复杂度最低的表达式
)

model.fit(x, y)
print("最佳公式:", model.sympy())