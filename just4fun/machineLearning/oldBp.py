import math
import time

import numpy
import funcSet

# 问题就在这里，应当在学习完成后，观察权值如何变化，主要是self.b如何变化？？
# 人类思考几道同类型的问题后就可以总结出这里问题的相似点，机器也一定可以，因为人也是物质组成的
# 如何看出问题是否是同一类问题？这样可以把经验值代入，经验模型带入，这样学习更快，缩短求解时间，经验模型包含，节点数，层数，激活函数，节点值？？
# 切记要控制变量法，不能乱来，否则大脑都会混乱，只有线性关系才是真正的规律，同类，抽象到n层后，成线性关系，可以认为是同类
# 控制变量法，先确定节点数，层数，激活函数，比较学习率的影响，然后比较节点数的影响，再比较层数的影响，再比较激活函数的影响？？？
# 人类的记忆能力可以训练，只要把权值置为100就行


def getLoss(errOut, corOut):
    return numpy.sum(1 / 2 * (errOut - corOut) ** 2)


class backpropagation:
    def __init__(self, hideLen, studyRate, activeFunc):
        self.inToHide = numpy.random.randn(len(inputs[0]), hideLen)
        # self.inToHide = numpy.zeros((len(inputs[0]), hideLen))
        self.hideToOut = numpy.random.randn(hideLen, len(outputs[0]))
        # self.hideToOut = numpy.zeros((hideLen, len(outputs[0])))
        self.b = numpy.random.randn(hideLen)
        # self.b = numpy.zeros(hideLen)
        self.studyRate = studyRate
        self.activeFunc = activeFunc

    def startTrain(self):
        global y, cost
        oldCost = 0
        for y in range(10000):
            cost = self.singleTrain()
            # cost = self.setTrain()
            # setTrain不行！！
            if 0 < oldCost < cost:
                print('cost > oldCost！！！' + str(oldCost))
                break
            oldCost = cost
        print('cost: ' + str(cost))
        print("共循环" + str(y) + '次')
        print("-----------------------------")

    def printSelf(self):
        cost = 0
        for j in range(len(outputs)):
            tempHide = neural(inputs[j], self.inToHide, self.b, self.activeFunc.func)
            tempOut = numpy.dot([tempHide], self.hideToOut)[0]
            cost += getLoss(tempOut, outputs[j])
            print(tempOut)
        print('cost = ' + str(cost))
        print("-----------------------------")
        print(self.inToHide)
        print(self.hideToOut)
        print(self.b)
        print("-----------------------------")

    def setTrain(self):
        cost = 0
        tempInToHide = self.inToHide.copy()
        tempHideToOut = self.hideToOut.copy()
        tempB = self.b.copy()
        for i in range(len(outputs)):
            tempHide = neural(inputs[i], self.inToHide, self.b, self.activeFunc.func)
            tempOut = numpy.dot([tempHide], self.hideToOut)[0]
            bp(inputs[i], tempInToHide, tempHide, tempHideToOut, tempOut, outputs[i], self.studyRate/len(inputs), tempB,
               self.activeFunc.derivativeFunc)
            cost += getLoss(tempOut, outputs[i])
        self.inToHide = tempInToHide.copy()
        self.hideToOut = tempHideToOut.copy()
        self.b = tempB.copy()
        return cost

    def singleTrain(self):
        cost = 0
        for i in range(len(outputs)):
            tempHide = neural(inputs[i], self.inToHide, self.b, self.activeFunc.func)
            tempOut = numpy.dot([tempHide], self.hideToOut)[0]
            bp(inputs[i], self.inToHide, tempHide, self.hideToOut, tempOut, outputs[i], self.studyRate/len(inputs), self.b,
               self.activeFunc.derivativeFunc)
            cost += getLoss(tempOut, outputs[i])
        return cost

    def test(self, inp):
        outs = numpy.ndarray((len(inp), len(inp[0])))
        for i in range(len(inp)):
            tempHide = neural(inp[i], self.inToHide, self.b, self.activeFunc.func)
            outs[i] = numpy.dot([tempHide], self.hideToOut)[0]
        return outs


def neural(inp, inToHide, inherent, func):
    tempHide = numpy.dot(inp, inToHide)
    tempHide += inherent
    # 神经元固有值，也可以优化，暂时不动
    tempHide = func(tempHide)
    return tempHide


def bp(inp, inToHide, hide, hideToOut, errOut, corOut, studyRate, b, derivativeFunc):
    d_loss_o = errOut - corOut

    d_loss_ho = numpy.sum(hideToOut * d_loss_o, axis=1)

    bpForOneLayer(inp, inToHide, d_loss_ho, hide, studyRate, b, derivativeFunc)

    d_loss_hToOut = numpy.dot(hide.reshape(len(hide), 1), d_loss_o.reshape(1, len(d_loss_o)))

    hideToOut -= studyRate * d_loss_hToOut


def bpForOneLayer(inp, inToHide, d_loss_ho, ho, studyRate, b, derivativeFunc):
    d_ho_h = derivativeFunc(ho)

    d_loss_h = d_loss_ho * d_ho_h
    # len(d_loss_h) = len(h)
    b -= studyRate * d_loss_h

    d_loss_inToH = numpy.dot(inp.reshape(len(inp), 1), d_loss_h.reshape(1, len(d_loss_h)))
    # d_loss_a.shape = inToHide.shape = (len(inp), len(h))

    inToHide -= studyRate * d_loss_inToH


def studyRateTest(funcGroup, hideLen):
    for i in range(1, 2):
        studyRate = 0.2 * i
        print('studyRate = %f' % studyRate)
        shit = backpropagation(hideLen, studyRate, funcGroup)
        shit.startTrain()
        print(shit.test(inputs))
        time.sleep(1)


def funcTest():
    for fs in funcSet.allBackAbleSets:
        studyRate = 0.8 / len(inputs)**2
        print(studyRate)
        print('funcSet = %s' % fs.name)
        shit = backpropagation(10, studyRate, fs)
        shit.startTrain()


def hideLenTest():
    for j in range(1, 10):
        studyRate = 0.8 / (11 - j) / (11 - j)
        outputs = numpy.array(outputs[0:-1])
        inputs = numpy.array(inputs[0:-1])
        print(inputs)
        shit = backpropagation(11 - j, studyRate, funcSet.tanhSet)
        shit.startTrain()


if __name__ == '__main__':
    # inputs = numpy.array([[0, 2], [1, 3], [-2, 0], [3, 5], [-4, -2], [5, 7], [-6, -4], [7, 9], [-8, -6], [9, 11]])
    # outputs = numpy.array([[0, 2], [1, 3], [-2, 0], [3, 5], [-4, -2], [5, 7], [-6, -4], [7, 9], [-8, -6], [9, 11]])
    inputs = numpy.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    outputs = numpy.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    studyRateTest(funcSet.reLuSet, 2)
'''
正向计算：
h1 = i1*a11 + i2*a21 + b1
ho1 = sigmoid(h1)
o1 = ho1 * b11 + ho2 * b21
o2 = ho1 * b12 + ho2 * b22
loss = 1/2(O1 - o1)^2 + 1/2(O2 - o2)^2

计算偏导数，用于更新权值和固定值：
$(a, b)表示a对b的偏导数
$(loss, o1) = o1 - O1
$(loss, o2) = o2 - O2 
$(o1, b11) = ho1
$(o2, b11) = ho2
$(o3, b11) = ho3
$(o4, b11) = ho4
    $(loss, b11) = $(loss, o1) * $(o1, b11) = (o1 - O1) * ho1   
    $(loss, b12) = (o2 - O2) * ho1    
    $(loss, b21) = (o1 - O1) * ho2
    $(loss, b22) = (o2 - O2) * ho2
$(o1, ho1) = b11
$(o2, ho1) = b12
$(loss, ho1) = $(loss, o1) * $(o1, ho1) + $(loss, o2) * $(o2, ho1) = (o1 - O1) * b11 + (o2 - O2) * b12
$(loss, ho2) = $(loss, o1) * $(o1, ho2) + $(loss, o2) * $(o2, ho2) = (o1 - O1) * b21 + (o2 - O2) * b22
$(ho1, h1) = ho1 - ho1^2
$(ho2, h2) = ho2 - ho2^2
$(loss, h1) = [(o1 - O1) * b11 + (o2 - O2) * b12] * (ho1 - ho1^2)
$(loss, h2) = [(o1 - O1) * b21 + (o2 - O2) * b22] * (ho2 - ho2^2)

更新w：
$(h1, a11) = i1
    $(loss, a11) = $(loss, h1) * $(h1, a11) = [(o1 - O1) * b11 + (o2 - O2) * b12] * (ho1 - ho1^2) * i1
    $(loss, a21) = [(o1 - O1) * b11 + (o2 - O2) * b12] * (ho1 - ho1^2) * i2
    $(loss, a12) = [(o1 - O1) * b21 + (o2 - O2) * b22] * (ho2 - ho2^2) * i1
    $(loss, a22) = [(o1 - O1) * b21 + (o2 - O2) * b22] * (ho2 - ho2^2) * i2

更新b：
$(loss, b1) = $(loss, h) * $(h, b1) = $(loss, h1)
'''