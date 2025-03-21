import math

import matplotlib.pyplot as plt
import numpy
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


if __name__ == '__main__':
    num_big = 10000
    arr1 = numpy.random.randn(num_big)*2+1
    arr2 = numpy.random.randn(num_big)+1
    print(numpy.std(arr1), numpy.average(arr1))
    print(numpy.std(arr2), numpy.average(arr2))
    arr3 = arr1*arr2
    print(numpy.std(arr3), numpy.average(arr3))
    print('-----------------------')
    random1 = np.random.uniform(1, 2, num_big) * 8 - 11
    random2 = np.random.uniform(1, 2, num_big) * 4 - 5
    print(numpy.std(random1), numpy.average(random1))
    print(numpy.std(random2), numpy.average(random2))
    random3 = random1*random2
    print(numpy.std(random3), numpy.average(random3))
    # say_hello()

    # drawGuss()