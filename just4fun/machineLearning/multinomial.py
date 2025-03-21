import numpy
from just4fun.machineLearning import funcSet

from just4fun.machineLearning import neuralNetwork


class MultinomialLine:
    def __init__(self, inputLen, outputLen, activeFuncSet, lr=0.001):
        self.inputs = None
        self.neuralLine1 = neuralNetwork.NeuralLine(inputLen, outputLen, funcSet.straightSet, lr, useBias=False)
        self.neuralLine2 = neuralNetwork.NeuralLine(inputLen, outputLen, funcSet.straightSet, lr, useBias=False)
        self.neuralLine3 = neuralNetwork.NeuralLine(inputLen, outputLen, funcSet.straightSet, lr, useBias=True)
        self.funcSet = activeFuncSet
        self.outputs = None
        self.next = None
        self.pre = None
        self.lr = lr

    def input(self, inputArr):
        self.inputs = inputArr
        self.neuralLine1.input(inputArr)
        self.neuralLine2.input(inputArr)
        self.neuralLine3.input(inputArr)
        self.outputs = self.funcSet.func(self.neuralLine1.outputs * self.neuralLine2.outputs + self.neuralLine3.outputs)

    def backpropagation(self, d_loss_ho):
        d_ho_h = self.funcSet.derivativeFunc(self.outputs)
        d_loss_h = d_loss_ho * d_ho_h
        b3 = self.neuralLine3.backpropagation(d_loss_h)
        b2 = self.neuralLine2.backpropagation(d_loss_h * self.neuralLine1.outputs)
        b1 = self.neuralLine1.backpropagation(d_loss_h * self.neuralLine2.outputs)
        #  y = ax*bx+cx     y' = 2abx + c
        return 2 * b1 * b2 * self.inputs + b3

    def update(self, steps, optim):
        self.neuralLine1.update(self.inputs, steps, optim)
        self.neuralLine2.update(self.inputs, steps, optim)
        self.neuralLine3.update(self.inputs, steps, optim)

    def updateStudyRate(self, rate):
        self.lr *= rate
        self.neuralLine1.studyRate = self.lr
        self.neuralLine2.studyRate = self.lr
        self.neuralLine3.studyRate = self.lr


class MultinomialNet:
    def __init__(self, inputLen, neuralLenArray, funcArray, studyRateArray):
        self.netHead = MultinomialLine(inputLen, neuralLenArray[0], funcArray[0], studyRateArray[0])
        self.netEnd = self.netHead
        self.step = 0
        self.costRecord = numpy.zeros(5)  # 判断cost进步是否太小
        self.oldCost = -1
        for i in range(1, len(neuralLenArray)):
            self.netEnd.next = MultinomialLine(neuralLenArray[i - 1], neuralLenArray[i], funcArray[i],
                                               studyRateArray[i])
            self.netEnd.next.pre = self.netEnd
            self.netEnd = self.netEnd.next

    def printSelf(self):
        print('————————————————————————————')
        temp = self.netHead
        while temp is not None:
            temp.neuralLine1.printSelf()
            temp.neuralLine2.printSelf()
            temp.neuralLine3.printSelf()
            print('-----------------------')
            temp = temp.next
        print('————————————————————————————')

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
            temp.updateStudyRate(rate)
            temp = temp.next

    def seeCostAndChangeRate(self, outputs, changeRate=0.8):
        cost = neuralNetwork.getCost(self.netEnd.outputs, outputs)
        if 0 < self.oldCost < cost:
            # 说明神经网络结构不对，学习速度不对，激活函数不对
            self.updateStudyRate(changeRate)
            print('oldCost < cost!!!', self.oldCost, ': ', cost, '  step: ', self.step, '  studyRate: ',
                  self.netHead.lr)
        else:
            if self.netHead.lr < 1:
                self.updateStudyRate(1.01)
        if self.step % len(self.costRecord) == len(self.costRecord) - 1 and \
                self.costRecord[0] - 1.1 * self.costRecord[-1] < 0:
            if self.netHead.lr < 1:
                self.updateStudyRate(1.5)
        self.oldCost = cost

    def forwardAndBack(self, inputs, outputs, optimize=''):
        self.input(inputs)
        # print(self.netEnd.outputs)
        d_loss_o = neuralNetwork.get_d_loss_o(self.netEnd.outputs, outputs)
        temp = self.netEnd
        d_loss_o = temp.backpropagation(d_loss_o)
        temp = temp.pre
        while temp is not None:
            d_loss_o = temp.backpropagation(d_loss_o)
            temp = temp.pre
        temp = self.netHead
        temp.update(self.step, optimize)
        temp = temp.next
        while temp is not None:
            temp.update(self.step, None)
            temp = temp.next

    def train(self, inputs, outputs, times=1000):
        while self.step < times:
            self.step += 1
            self.costRecord[self.step % len(self.costRecord)] = self.oldCost
            self.forwardAndBack(inputs, outputs, optimize='')
            self.seeCostAndChangeRate(outputs, changeRate=0.8)
            self.costRecord[self.step % len(self.costRecord)] = self.oldCost
            if self.oldCost < 1e-5 or self.netEnd.lr < 1e-6:
                print('ending: cost = %f, studyRate=%f step=%d' % (self.oldCost, self.netEnd.lr, self.step))
                return


def test1():
    # inputs = numpy.array([[1, 0], [0, 1], [1, 1], [0, 0], [2, 1], [2, 1], [1, 2], [2, 2]])
    inputs = numpy.random.random((32, 4)) * 4 - 2
    # outputs = numpy.array([[0], [0], [1], [1], [2], [2], [1], [1]])
    outputs = 0.7 * inputs ** 2 - 3 * inputs + 4
    print(inputs, outputs)
    # outputs = numpy.array([[3], [1], [54], [-20], [-115], [20]])
    neuralLenArray = numpy.array([len(outputs[0])])
    funcArray = [funcSet.straightSet, funcSet.straightSet, funcSet.straightSet]  # 输出有0，1不能用relu? 因为0不进行反向传播。。
    # studyRateArray = numpy.array([0.8, 0.1], dtype=numpy.float32) / (neuralLenArray ** 2)
    studyRateArray = numpy.array([0.001, 0.001], dtype=numpy.float32)  # 学习速度非常重要，过高过低都不行。
    # print(studyRateArray._FloatType)
    print(studyRateArray)
    nnet = MultinomialNet(len(inputs[0]), neuralLenArray, funcArray, studyRateArray)
    nnet.printSelf()
    # nnet.batchTrain(inputs, outputs, batchSize=1, times=10000)
    nnet.train(inputs, outputs, times=1000)
    print('self.step = %d, cost = %f, studyRate = %f' % (nnet.step, nnet.oldCost, nnet.netEnd.lr))
    nnet.printSelf()
    # print(nnet.netEnd.outputs)
    # print(outputs)
    # print(neuralNetwork.getCost(nnet.netEnd.outputs, outputs))

    print(neuralNetwork.getCost(nnet.netEnd.outputs, outputs))


if __name__ == '__main__':
    test1()
