import math

import matplotlib.pyplot as plt
import numpy
import pandas
from PIL import Image
from numpy import clip


def pictureGenerate():
    img = Image.open('E:\PythonProjects\pictures\截图1.png', 'r')
    print(img)
    outputPath = 'E:/PythonProjects/pictures/'
    arr = numpy.array(img)
    print(type(arr))
    print(arr.shape)
    k = 50
    u_r, sigma_r, v_r = numpy.linalg.svd(arr[:, :, 0])
    u_g, sigma_g, v_g = numpy.linalg.svd(arr[:, :, 1])
    u_b, sigma_b, v_b = numpy.linalg.svd(arr[:, :, 2])
    plt.figure(figsize=(8, 8), facecolor='w')


def numpy2():
    d = numpy.random.rand(3, 4)
    print(d)
    print(type(d))
    data = pandas.DataFrame(data=d, columns=list('dcba'))
    print(data)
    print(type(data))
    print(data[list('bd')])
    data.to_csv('data.csv', index=False, header=True)


def numpy3():
    print(numpy.sqrt(numpy.sum(6 / numpy.arange(1, 10000) ** 2)))
    print(numpy.sqrt(6 * numpy.sum(1 / numpy.arange(1, 10000) ** 2)))


def numpy4():
    x = numpy.linspace(0, 2, 100)
    print(x)
    y = x ** x
    plt.plot(x, y, 'r-', linewidth=2)
    plt.show()
    plt.plot(x, y, 'r:', linewidth=2)
    plt.show()
    plt.plot(x, y, 'r.', linewidth=2)
    plt.show()


def numpy5():
    frequency = numpy.zeros(9, dtype=int)
    print(frequency)
    n = 1
    frequency.sort()
    for i in range(1, 10000):
        n *= i
        while n >= 100000000:
            n //= 100000000
        while n >= 10:
            n //= 10
        frequency[n - 1] += 1
    plt.plot(frequency, 'r-', linewidth=2)
    plt.plot(frequency, 'go', markersize=8)
    plt.title('%d!首位数字出现频率' % 1000, fontsize=18)
    plt.grid(b=True, ls=':', color='#00ffaa')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.show()


def drawSth3():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    path = 5000
    n = 10
    v0 = 5
    p = 0.1
    Times = 3000
    numpy.random.seed(0)
    x = numpy.random.rand(n) * path
    # print(x)
    x.sort()
    v = numpy.tile([v0], n).astype(float)
    print(v)

    plt.figure(figsize=(10, 8), facecolor='w')  #10*8英寸，白色背景
    for t in range(Times):
        plt.scatter(x, [t]*n, s=1, c='k', alpha=0.05)
        for i in range(n):
            if x[(i + 1) % n] > x[i]:
                d = x[(i + 1) % n] - x[i]
            else:
                d = path - x[i] + x[(i + 1) % n]
            if v[i] < d:
                if numpy.random.rand() > p:
                    v[i] += 1
                else:
                    v[i] -= 1
            else:
                v[i] = d - 1
        v = v.clip(0, 150)
        x += v
        clip(x, a_min=0, a_max=path)
    plt.xlim(0, path)
    plt.ylim(0, Times)
    plt.xlabel('车辆位置', fontsize=16)
    plt.ylabel('模拟时间', fontsize=16)
    plt.title('环形公路堵车模拟', fontsize=16)
    plt.tight_layout(pad=2)
    # plt.savefig('result.png')
    plt.show()


def funcToPic():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    p = numpy.linspace(0.01, 0.99, 100)
    h = -(1-p)*numpy.log2(1-p) - p*numpy.log2(p)
    plt.plot(p, h/2, 'r-', linewidth=3)
    plt.xlabel('概率', fontsize=16)
    plt.ylabel('熵值', fontsize=16)
    # plt.title('entropy', fontsize=18)
    h = p * (1 - p)
    plt.plot(p, h*2, 'r-', linewidth=3)
    plt.show()


def numpy7():
    data = numpy.random.rand(10, 3)
    x = data[:, [0, 1]]
    print(x)
    print(data)


def drawXYFuc(string, ranges):
    x = numpy.linspace(-ranges, ranges, ranges * 20)
    y = eval(string)
    plt.plot(x, y, 'r-')
    plt.show()


def sigmoid(x):
    return 1 / (1 + numpy.e ** -x)


def tanh(x):
    a = math.e ** x
    b = math.e ** -x
    return (a - b) / (a + b)


def circle(r):
    x = numpy.linspace(-r, r, 100)
    y1 = (r**2 - x**2) ** (1/2)
    y2 = -y1
    return [x, x], [y1, y2]


def plotlyDraw(string):
    x = numpy.linspace(-5, 5, 100)
    print(x)
    y = eval(string)
    print(y)
    plt.plot(x, y, 'b.')
    plt.show()


def drawIm():
    r = numpy.tile(numpy.linspace(192, 255, 300, dtype=numpy.uint8), (600, 1)).T
    g = numpy.tile(numpy.linspace(192, 255, 600, dtype=numpy.uint8), (300, 1))
    b = numpy.ones((300, 600), dtype=numpy.uint8) * 224
    im = numpy.dstack((r, g, b))
    x = numpy.arange(600)
    y = numpy.sin(numpy.linspace(0, 2 * numpy.pi, 600))
    print(y)
    y = numpy.int32((y + 1) * 0.9 * 300 / 2 + 0.05 * 300)
    for i in range(0, 150, 6):
        im[y[:-i], (x + i)[:-i]] = numpy.array([255, 0, 255])

    Image.fromarray(im, mode='RGB').show()

    drawXYFuc('(1.1 ** x - 0.9 ** x) * 50', 20)
    x, y = circle(2)
    plt.plot(x, y, 'r.')
    plt.show()

    im = numpy.random.randint(0, 255, (400, 1200, 3), dtype=numpy.uint8)
    im = Image.fromarray(im, mode='RGB')
    im.show()  # 或者im.save(r'd:\gray_300_100.jpg')保存为文件


if __name__ == '__main__':
    pictureGenerate()
