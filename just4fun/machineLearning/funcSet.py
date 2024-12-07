import math

import numpy


class FuncSet:
    def __init__(self, func, derivativeFunc):
        self.name = str(func)
        self.func = func
        self.derivativeFunc = derivativeFunc


def d_sigmoid(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + math.e ** -x)


def tanh(x):
    x = numpy.minimum(x, 100)
    x = numpy.maximum(x, -100)
    a = math.e ** x
    b = math.e ** -x
    return (a - b) / (a + b)


def d_tanh(x):
    return 1 - x ** 2


def straight(x):
    return x.copy()


def d_straight(x):
    return 1


# def square(x):
#     return x ** 2
# 平方立方需要记住因变量，不然不好求导，不知道导数是正还是负数。。。立方呢？？
# 很显然，激活函数和


def powE(x):
    return math.e ** x


def d_powE(x):
    return x * 1


def reLu(x):
    return numpy.maximum(0, x)


def d_reLu(x):
    return (x > 0) * 1


def reLu2(x):
    return numpy.maximum(0, x) + numpy.minimum(0, x) / 8


def d_reLu2(x):
    return (x > 0) * 1 + (x < 0) / 8


def reLuTanh(x):
    return numpy.maximum(0, x) + tanh(numpy.minimum(0, x))


def d_reLuTanh(x):
    return d_tanh((x < 0) * x)


sigmoidSet = FuncSet(sigmoid, d_sigmoid)
straightSet = FuncSet(straight, d_straight)
tanhSet = FuncSet(tanh, d_tanh)
reLuSet = FuncSet(reLu, d_reLu)
reLu2Set = FuncSet(reLu2, d_reLu2)
reLuTanhSet = FuncSet(reLuTanh, d_reLuTanh)
powESet = FuncSet(powE, d_powE)
allFuncSets = [straightSet, reLuSet, tanhSet, powESet]


def printArr(arr):
    outputs = '['
    for arr in arr:
        outputs += '['
        for i in arr:
            outputs += str(i) + ', '
        outputs = outputs[0:-2]
        outputs += '], '
    outputs = outputs[0:-2]
    outputs += ']'
    print(outputs)


def generateTrainSet(x, y, z):
    arr = numpy.random.rand(x, y) * 2 * z - z
    arr1 = numpy.random.rand(x, y) * 2 * z - z
    printArr(arr)
    printArr(arr1)
    return arr, arr1