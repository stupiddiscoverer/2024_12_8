import numpy

from just4fun.machineLearning import funcSet
#     # y = |x - c|   y' = 1 (x-c>0), -1 (x-c<0)
#     return converted


def get_loss(errOut, corOut):
    # return errOut - corOut
    return numpy.sum(numpy.abs(errOut - corOut)) / len(errOut)


def getCost(errOut, corOut):
    return numpy.sum(1 / 2 * (errOut - corOut) ** 2) / len(errOut)


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

    def printSelf(self):
        print(self.matrix)
        print(self.bias)
        print(self.activeFunc.name)

    def input(self, inputs):
        self.outputs = numpy.dot(inputs, self.matrix)
        if self.bias is not None:
            self.outputs += self.bias
        self.outputs = self.activeFunc.func(self.outputs)

    def update(self, inputs, loss):
        self.d_loss_h = loss * self.outputs * self.studyRate
        if self.bias is not None:
            db = numpy.sum(self.d_loss_h, axis=0, keepdims=True) / len(self.outputs)
            self.bias -= db
        dw = numpy.dot(inputs.T, self.d_loss_h)
        self.matrix -= dw


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

    def forwardAndBack(self, inputs, outputs):
        self.input(inputs)
        loss = getCost(self.netEnd.outputs, outputs)
        temp = self.netHead
        temp.update(inputs, loss)
        temp = temp.next
        while temp is not None:
            temp.update(temp.pre.outputs, loss)
            temp = temp.next

    def train(self, inputs, outputs, times=1000):
        while self.step < times:
            self.step += 1
            self.forwardAndBack(inputs, outputs)
            self.oldCost = getCost(self.netEnd.outputs, outputs)
            if self.oldCost < 1e-6 or self.netEnd.studyRate < 1e-7 or self.netHead.studyRate > 1:
                print('cost = %f, studyRate=%f' % (self.oldCost, self.netEnd.studyRate))
                return
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
    # inputs, outputs = selfMultiplyNet.generateTrainData()
    # outputs = numpy.array([[0], [0], [1], [1]])
    inputs = getInputs()
    outputs = getOutputs()
    neuralLenArray = numpy.array([10, len(outputs[0])])
    funcArray = [funcSet.reLu2Set, funcSet.reLu2Set, funcSet.straightSet]   # 输出有0，1不能用relu? 因为0不进行反向传播。。
    studyRateArray = numpy.array([0.01, 0.01, 0.01], dtype=numpy.float32)   # 学习速度非常重要，过高过低都不行。
    print(studyRateArray)
    nnet = NeuralNet(len(inputs[0]), neuralLenArray, funcArray, studyRateArray)
    nnet.printSelf()
    nnet.train(inputs, outputs)
    print('self.step = %d, cost = %f, studyRate = %f' % (nnet.step, nnet.oldCost, nnet.netEnd.studyRate))
    print(nnet.netEnd.outputs)

    print(outputs)
    print(getCost(nnet.netEnd.outputs, outputs))


if __name__ == '__main__':
    test()


