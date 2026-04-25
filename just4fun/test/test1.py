import numpy as np

# 固定随机种子
np.random.seed(42)

# 参数
n = 1000
mean_v = 1.0
std_v = 1.0

# 生成 V 的一列
v = np.random.normal(loc=mean_v, scale=std_v, size=n)

# 生成随机 logits，做 softmax
logits = np.random.randn(n)  # 标准正态分布
weights_softmax = np.exp(logits) / np.sum(np.exp(logits))  # 和 = 1

# 均匀权重
weights_uniform = np.ones(n) / n

# 加权平均
result_softmax = np.sum(weights_softmax * v)
result_uniform = np.sum(weights_uniform * v)

print("Softmax 权重（前10个）:", weights_softmax[:10].round(6))
print("最大权重:", weights_softmax.max().round(6))
print("最小非零权重:", weights_softmax[weights_softmax > 1e-10].min().round(6))
print()
print("V 均值:", v.mean().round(6))
print("V2 均值:", (v*v).mean().round(6))
print("V 标准差:", v.std().round(6))
print("V2 标准差:", (v*v).std().round(6))
print()
print("均匀权重结果:", result_uniform.round(6))
print("Softmax 权重结果:", result_softmax.round(6))