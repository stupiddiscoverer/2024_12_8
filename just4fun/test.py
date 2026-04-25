import numpy.random

import math

import matplotlib.pyplot as plt
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def drawGuss():
    data1 = np.random.rand(50000)
    data2 = np.random.rand(50000)
    print(np.std(data1), np.average(data1))
    data = (data1-0.5) * (data2-0.5) * 500 - 500
    for i in range(len(data)):
        data[i] = int(data[i])
    print(data)
    y = np.zeros(1000)
    for d in data:
        # print(x)
        y[int(d) + 500] += 1
    x = np.zeros(len(y))
    for i in range(len(x)):
        x[i] = i
    x -= 500
    plt.plot(x, y, label='10万个正态分布随机数', color='blue')
    # plt.show()

    y1 = 1/((2*math.pi)**0.5 * 100) * math.e**(-x**2/(2*100**2)) * 100000
    plt.plot(x, y1, label='高斯曲线', color='red')
    plt.show()


def selfMul(x, dim=1):
    len = x.shape[dim] // 2
    selfMul = x.narrow(dim, 0, len) * x.narrow(dim, len, len)
    x.narrow(dim, 0, len).copy_(selfMul)


def strangeStr():
    a = '𒀱𒀱𒀱𒀱𒀱𒀱𒀱𒀱𒀱𒀱'
    print(len(a))
    for c in a:
        print(ord(c), c)
    # 要编码的字符串
    utf8_encoded = a.encode('utf-8')
    print(len(utf8_encoded))
    utf8_code_points = list(utf8_encoded)
    print(len(utf8_code_points))
    print(f"The UTF-8 code are {utf8_code_points}")


def randomTest():
    print(np.dot(np.random.rand(1, 10000) * 2 - 1, np.random.rand(10000, 3) * 2 - 1))
    print(np.dot(np.random.rand(1, 1000) * 2 - 1, np.random.rand(1000, 3) * 2 - 1) / 1000 ** 0.5)
    print(np.dot(np.random.rand(1, 100) * 2 - 1, np.random.rand(100, 3) * 2 - 1) / 100 ** 0.5)
    print(np.dot(np.random.rand(1, 10) * 2 - 1, np.random.rand(10, 3) * 2 - 1) / 10 ** 0.5)

from Crypto.Cipher import AES


def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result
    return wrapper


@my_decorator
def say_hello():
    print("Hello!")


def decrypt_aes_128_cbc(key, iv, ciphertext):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)
    return decrypted.rstrip(b"\0")  # 去掉填充的 \0

import torch

def visualize_computation_graph():
    """
    正确的 detach() 行为演示
    """
    # 创建计算图
    a = torch.tensor([2.0], requires_grad=True)
    b = a * 3
    c = b ** 2
    d = c.detach()  # d 从计算图中分离，不再需要梯度
    e = d + 5  # e 基于 d 创建，d 不需要梯度，所以 e 也不需要梯度

    print(f"a.requires_grad: {a.requires_grad}")  # True
    print(f"b.requires_grad: {b.requires_grad}")  # True
    print(f"c.requires_grad: {c.requires_grad}")  # True
    print(f"d.requires_grad: {d.requires_grad}")  # False ← detach 后不再需要梯度
    print(f"e.requires_grad: {e.requires_grad}")  # False ← 继承自 d

    print(f"c.grad_fn: {c.grad_fn}")  # 有 grad_fn (PowBackward0)
    print(f"d.grad_fn: {d.grad_fn}")  # None ← detach 后 grad_fn 为 None
    print(f"e.grad_fn: {e.grad_fn}")  # None ← 因为没有梯度追踪

    # 尝试反向传播
    try:
        e.sum().backward()  # 这会失败，因为 e 不需要梯度
    except RuntimeError as e:
        print(f"错误: {e}")


def explain_lbfgs_math():
    """
    解释 LBFGS 的数学原理
    """
    print("LBFGS 的核心思想:")
    print("=" * 50)

    print("1. 梯度下降 (SGD):")
    print("   x_{k+1} = x_k - α * ∇f(x_k)")
    print("   只用一阶信息，收敛慢")

    print("\n2. 牛顿法:")
    print("   x_{k+1} = x_k - [∇²f(x_k)]⁻¹ * ∇f(x_k)")
    print("   用海森矩阵，收敛快但计算量大 O(n³)")

    print("\n3. LBFGS (折中方案):")
    print("   • 用过去 m 步的梯度信息近似海森矩阵")
    print("   • 存储量 O(mn) 而不是 O(n²)")
    print("   • 计算量 O(mn) 而不是 O(n³)")
    print("   • 既快又省内存")


def derivative_xx_method1():
    """
    方法1：对数求导法
    """
    print("令 y = x^x")
    print("\n步骤1：两边取自然对数")
    print("  ln y = ln(x^x) = x * ln x")

    print("\n步骤2：两边对x求导")
    print("  (1/y) * y' = ln x + x * (1/x)")
    print("  (1/y) * y' = ln x + 1")

    print("\n步骤3：解出 y'")
    print("  y' = y * (ln x + 1)")
    print("  y' = x^x * (ln x + 1)")

    print("\n最终结果：")
    print("  d/dx (x^x) = x^x * (ln x + 1)")


if __name__ == '__main__':
    visualize_computation_graph()
    # explain_lbfgs_math()
    # derivative_xx_method1()
    # print(help(range))