import math

import numpy

from machineLearning import funcSet
from machineLearning import selfMultiplyNet


# def get_d_loss_o_abs(errOut, corOut):
#     # y = |x - c|   y' = 1 (x-c>0), -1 (x-c<0)
#     converted = [[1 if x > 0 else -1 if x < 0 else 0 for x in row] for row in (errOut - corOut)]
#     return converted


def get_d_loss_o(errOut, corOut):
    return errOut - corOut


# def getCost_abs(errOut, corOut):
#     return numpy.sum(numpy.abs(errOut - corOut)) / len(errOut)


def getCost(errOut, corOut):
    return numpy.sum(1 / 2 * (errOut - corOut) ** 2) / len(errOut)


def disperseMatrix(updateMatrix, interval=0.01):
    # randomUp = numpy.random.random_integers(-1, high=1, size=updateMatrix.shape) * interval
    # updateMatrix = updateMatrix / interval
    # updateMatrix = updateMatrix.astype(dtype=numpy.int32)
    # updateMatrix = updateMatrix.astype(dtype=numpy.float32)
    # return updateMatrix * interval + randomUp
    return updateMatrix


class NeuralLine:
    def __init__(self, inputLen, hiddenLen, activeFunc, studyRate, useBias=True):
        self.inputLen = inputLen
        self.matrix = numpy.random.rand(inputLen, hiddenLen) * 2 - 1
        self.outputs = None
        self.bias = numpy.random.rand(1, hiddenLen) * 2 - 1 if useBias else None
        self.activeFunc = activeFunc
        self.studyRate = studyRate
        self.next = None
        self.pre = None
        self.d_loss_h = None
        # 以下用于实现adam优化算法
        self.vDw = numpy.zeros((inputLen, hiddenLen))
        self.sDw = numpy.zeros((inputLen, hiddenLen))
        self.vDb = numpy.zeros((1, hiddenLen))
        self.sDb = numpy.zeros((1, hiddenLen))
        self.lastDw = None
        self.lastDb = None

    def printSelf(self):
        print(self.matrix)
        print(self.bias)
        print(self.activeFunc.name)
        # print(self.studyRate)

    def input(self, inputs):
        self.outputs = numpy.dot(inputs, self.matrix)
        if self.bias is not None:
            self.outputs += self.bias
        self.outputs = self.activeFunc.func(self.outputs)

    def adamUpdate(self, inputs, step):
        db = numpy.sum(self.d_loss_h, axis=0, keepdims=True)
        self.vDb = 0.9 * self.vDb + 0.1 * db
        self.sDb = 0.9 * self.sDb + 0.1 * db ** 2

        cor = 1 / numpy.sqrt(1 - 0.9 ** step)

        if self.bias is not None:
            self.lastDb = self.studyRate * cor * self.vDb / numpy.sqrt(self.sDb + 1e-10) * 0.03
            self.bias -= self.lastDb

        # 这里比较烦，不过相乘后矩阵行列数对了，正好就对了
        dw = numpy.dot(inputs.T, self.d_loss_h)
        dw = disperseMatrix(dw)

        self.vDw = 0.9 * self.vDw + 0.1 * dw
        self.sDw = 0.9 * self.sDw + 0.1 * dw ** 2

        self.lastDw = self.studyRate * cor * self.vDw / numpy.sqrt(self.sDw + 1e-10)
        self.matrix -= self.lastDw

    def unUpdate(self):
        self.matrix += self.lastDw * 0.9
        if self.bias is not None:
            self.bias += self.lastDb

    def update(self, inputs, step=0, optimize='adam'):
        if optimize == 'adam':
            return self.adamUpdate(inputs, step)

        if self.bias is not None:
            db = numpy.sum(self.d_loss_h, axis=0, keepdims=True) / len(inputs)
            self.lastDb = self.studyRate * db
            self.bias -= self.lastDb

        # 这里比较烦，不过相乘后矩阵行列数对了，正好就对了
        dw = numpy.dot(inputs.T, self.d_loss_h) / self.inputLen**0.5 / len(inputs)
        dw = disperseMatrix(dw)
        self.lastDw = self.studyRate * dw
        self.matrix -= self.lastDw

    def backpropagation(self, d_loss_ho):
        d_ho_h = self.activeFunc.derivativeFunc(self.outputs)

        self.d_loss_h = d_loss_ho * d_ho_h

        # 这里也很烦，不过相乘后矩阵行列数对了，正好就对了
        # 我的妈，两种return都可以训练出结果！！！！而且效果差不多！！！是因为sigmoid和relu的导数变化不大，乘了和没乘差不多！！tanh就不行
        return numpy.dot(self.d_loss_h, self.matrix.T)


class NeuralNet:
    def __init__(self, inputLen, neuralLenArray, funcArray, studyRateArray):
        self.netHead = NeuralLine(inputLen, neuralLenArray[0], funcArray[0], studyRateArray[0])
        self.netEnd = self.netHead
        self.step = 0
        self.oldCost = 0
        self.costRecord = numpy.zeros(5)
        for i in range(1, len(neuralLenArray)):
            self.netEnd.next = NeuralLine(neuralLenArray[i - 1], neuralLenArray[i], funcArray[i], studyRateArray[i])
            self.netEnd.next.pre = self.netEnd
            self.netEnd = self.netEnd.next

    def printSelf(self):
        print('///////////////////////////')
        temp = self.netHead
        while temp is not None:
            print(temp.matrix)
            print(temp.bias)
            print(temp.studyRate)
            print('-----------------------')
            temp = temp.next
        print('///////////////////////////')

    def input(self, inputs):
        temp = self.netHead
        temp.input(inputs)
        temp = temp.next
        while temp is not None:
            temp.input(temp.pre.outputs)
            temp = temp.next

    def updateStudyRate(self, rate):
        temp = self.netHead
        while temp is not None:
            temp.studyRate *= rate
            temp = temp.next

    def unUpdate(self):
        temp = self.netHead
        while temp is not None:
            temp.unUpdate()
            temp = temp.next

    def seeCostAndChangeRate(self, outputs, changeRate=0.8):
        cost = getCost(self.netEnd.outputs, outputs)
        if 0 < self.oldCost < cost:
            # 说明神经网络结构不对，学习速度不对，激活函数不对
            # self.unUpdate()
            self.updateStudyRate(changeRate)
            print('oldCost < cost!!!', self.oldCost, ': ', cost, '  step: ', self.step, '  studyRate: ',
                  self.netHead.studyRate)
        else:
            if self.netHead.studyRate < 1:
                self.updateStudyRate(1.01)
            if self.step % len(self.costRecord) == len(self.costRecord) - 1 and \
                    self.costRecord[0] - 1.01*self.costRecord[-1] < 0:
                if self.netHead.studyRate < 1:
                    self.updateStudyRate(2)
        self.oldCost = cost

    def forwardAndBack(self, inputs, outputs, optimize=''):
        self.input(inputs)
        d_loss_o = get_d_loss_o(self.netEnd.outputs, outputs)
        temp = self.netEnd
        d_loss_o = temp.backpropagation(d_loss_o)
        temp = temp.pre
        while temp is not None:
            d_loss_o = temp.backpropagation(d_loss_o)
            temp = temp.pre
        temp = self.netHead
        temp.update(inputs, self.step, optimize)
        temp = temp.next
        while temp is not None:
            temp.update(temp.pre.outputs, self.step)
            temp = temp.next

    def train(self, inputs, outputs, times=1000):
        while self.step < times:
            self.step += 1
            self.forwardAndBack(inputs, outputs, optimize='')
            self.seeCostAndChangeRate(outputs, changeRate=0.8)
            if self.oldCost < 1e-6 or self.netEnd.studyRate < 1e-7 or self.netHead.studyRate > 1:
                print('cost = %f, studyRate=%f' % (self.oldCost, self.netEnd.studyRate))
                return

    def batchTrain(self, inputs, outputs, batchSize=4, times=1000):
        while self.step < times:
            index = 0
            while index < len(inputs):
                self.step += 1
                end = index + batchSize
                if end > len(inputs):
                    end = len(inputs)
                self.forwardAndBack(inputs[index:end], outputs[index:end], optimize='adam')
                index += batchSize
            self.input(inputs)
            self.seeCostAndChangeRate(outputs, changeRate=0.95)
            print(self.oldCost)
            if self.oldCost < 1e-6 or self.netEnd.studyRate < 0.00001:
                return
            self.oldCost = getCost(self.netEnd.outputs, outputs)

    def test(self, inputs):
        self.input(inputs)
        print(self.netEnd.outputs)


def getInputs():
    arr = numpy.random.random((10, 10)) / 10
    for i in range(10):
        arr[i][i] = 1.0
    return arr


def getOutputs():
    arr = numpy.zeros((10, 1))
    for i in range(10):
        arr[i][0] = i
    return arr


def test():
    # numpy.random.seed(103)
    # inputs, outputs = selfMultiplyNet.generateTrainData()
    # inputs = numpy.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    # outputs = numpy.array([[0], [0], [1], [1]])
    inputs = getInputs()
    outputs = getOutputs()
    # print(inputs, outputs)
    neuralLenArray = numpy.array([10, len(outputs[0])])
    funcArray = [funcSet.reLu2Set, funcSet.reLu2Set, funcSet.straightSet]   # 输出有0，1不能用relu? 因为0不进行反向传播。。
    studyRateArray = numpy.array([0.01, 0.01, 0.01], dtype=numpy.float32)   # 学习速度非常重要，过高过低都不行。
    print(studyRateArray)
    nnet = NeuralNet(len(inputs[0]), neuralLenArray, funcArray, studyRateArray)
    nnet.printSelf()
    # nnet.batchTrain(inputs, outputs, batchSize=1, times=10000)
    nnet.train(inputs, outputs)
    print('self.step = %d, cost = %f, studyRate = %f' % (nnet.step, nnet.oldCost, nnet.netEnd.studyRate))
    # nnet.printSelf()
    print(nnet.netEnd.outputs)

    print(outputs)
    print(getCost(nnet.netEnd.outputs, outputs))
    # nnet.test(numpy.array([[8, 1], [-1, -4], [9, -1], [1, -1], [111, 0]], dtype=numpy.float32))


# tanh是记忆随机数映射关系之神，可能是因为输入和输出都在-1到1之间
# leakyrelu也比relu更好训练，relu需要保证网络容量远大于需要记忆的数据可能2到3倍
# tanh表现比sigmoid好4倍

# 通过计算得知，学习速率要和inputs范围成反比，和outputs范围没啥关系！！！

# 我认为我很少有机会看这些注释，它们越来越多了，所以我还是记在脑子里吧。。。？
# 学习速度，激活函数，应当是每一个神经元自己学习还是每一层神经元自己学习？？
# 指数爆炸是否说明爆炸是指数形式的化学反应，那是原子裂变，链式反应
# 将每层权重控制在1/n附近，因为input在(-1,1),这样input*matrix得到的output也在(-1,1),也取决于activeFunc(output)
# x -= 1/m * sum(x), e2 = 1/m * sum(x^2), x /= e2. 记住1/m*sum(x) 和 e2，对output作同样处理

# momentum算法，或称为指数加权平均算法
# 每次更新权值的偏导数d_cost_w = 0.9 * pre_d_cost_w + 0.1 * cur_d_cost_w，只需要记住上一次偏导数即可

# 只有激活函数和潜在规律同级别才能通过实数集上的测试，比如指数模拟指数，平方模拟平方，如果激活函数不对，无论如何训练只是记住训练集范围内的映射关系
# 这时就要尝试新的激活函数。。
# 初始参数就像我们对事物的偏见，尝试不同的初始参数，比较优劣，留优去劣，就像别人和你说一个人怎么样，一件事怎么样，顺着别人的思路是否能得到更好的结果？？

if __name__ == '__main__':
    test()

# 一个重要的事情是，让程序尝试不同的超参数，尤其是激活函数的选择，甚至激活函数本身也可以用训练好的神经网络，
# 其实多层神经网络就是把神经网络当作激活函数，第4层把第3层当作第二层的输出的激活函数
# 多会一种题型，或多会一个激活函数是不是代表更聪明一点？
# 参数太多，必须让神经网络自己调整自己的参数，

# 人找规律都是线性的，人对看起来非线性的数据很反感
# 找规律本质是找某个抽象层的线性关系
# 对特定的数据模式只有几个待选的偏见算法，如果确定数据是有规律的，但已有的算法无法找到规律，可以认为这个数据就是一种规律？
# 而所谓线性又是什么？是我们倒背如流的一二三四五吗，我想一二三四五也不是天生的，也是学习来的规律吧？因为经常出现
# 人脑本身也自然随机进化而来，随机尝试的算法如果表现更优秀则可以留存下来，否则淘汰，就像模拟生命的起源和发展。。那有点慢
# 但最终，一定比现在的人类的思维方法更好，因为没有理由随机到完美的思维方法吧？即使是人也有笨的，或对某些问题笨。。
# ！！！！！！！！！！！！！！！！要遵循一切参数皆可训练原则！！！！！！！！！！！！！！！！
#   包括网络层数、层神经元数、连接方式、针对每个参数的学习速度
# ！！！！！！！！！！！！！！！！！！这样才是真正的智能，智能就是随机应变，适应环境！！！！！！！！！！！！！！！！！！

# 非线性函数无法模拟纯逻辑，只能模拟一定范围的输入对应的输出，刚好现实世界的输入都是一定范围的，比如像素值范围是0-255，常用动词只有30个？
