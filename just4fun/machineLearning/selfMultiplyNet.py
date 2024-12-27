import math
import random
import numpy


def get_d_loss_o(errOut, corOut):
    return errOut - corOut


def getCost(errOut, corOut):
    return numpy.sum(1 / 2 * (errOut - corOut) ** 2) / len(errOut)


class selfMultiplyLayer:
    def __init__(self, inputLen, studyRate=0.001):
        self.inputLen = inputLen
        self.studyRate = studyRate
        self.matrix = numpy.random.random((inputLen, inputLen)) * 2 - 1
        self.matrix2 = numpy.random.random((inputLen, inputLen)) * 2 - 1
        self.inputs = None
        self.outputs = None
        self.tempOut = None
        self.b = numpy.zeros((1, inputLen))
        self.cRate = 1
        self.bRate = 1

    def printSelf(self):
        print(self.matrix)
        print(self.b)
        print(self.studyRate)
        print(self.outputs)

    def input(self, inputs):
        self.inputs = inputs
        # y = x * (wx + b) + c = wx^2 + bx + c  足以模拟任意二次函数
        self.tempOut = numpy.dot(inputs, self.matrix)
        self.outputs = inputs * self.tempOut + self.b
        self.outputs = self.outputs / numpy.sqrt(self.inputLen)

    def backpropagation(self, d_loss_o):
        self.b -= numpy.sum(d_loss_o, axis=0)/len(self.inputs) * self.studyRate * self.cRate
        d_loss_tempOut = d_loss_o * self.inputs / numpy.sqrt(self.inputLen)
        self.b -= numpy.sum(d_loss_tempOut, axis=0)/len(self.inputs) * self.studyRate * self.bRate
        d_loss_in = self.tempOut * d_loss_o + numpy.dot(d_loss_tempOut, self.matrix.T)
        self.matrix -= numpy.dot(d_loss_tempOut.T, self.inputs) / len(self.inputs) * self.studyRate
        return d_loss_in


def getModelLayer(inputLen):
    model = selfMultiplyLayer(inputLen)
    for i in range(inputLen):
        for j in range(inputLen):
            model.matrix[i][j] = random.random() * 2 - 1
        model.b[0][i] = random.random() * 2 - 1
    return model


def generateInputs(inputLen=2):
    shapeIn = (3, inputLen)
    inputs = numpy.zeros(shapeIn)
    count = 0
    for i in range(shapeIn[0]):
        for j in range(inputLen):
            inputs[i][j] = (count // (4**j)) % 4
        count += 1
    inputs = inputs / 2 - 0.3
    return inputs


def generateTrainData(inputLen=2):
    inputs = generateInputs(inputLen)
    model = getModelLayer(inputLen)
    model.input(inputs)
    outputs = model.outputs
    return inputs, outputs


def toOne(matrix):
    sum1 = numpy.sum(matrix)
    average = sum1 / len(matrix) / len(matrix[0])
    matrix -= average
    squareDiff = numpy.sum(matrix ** 2)
    squareDiff = squareDiff / len(matrix) / len(matrix[0])
    squareDiff = squareDiff ** 0.5
    matrix = matrix / squareDiff
    return average, squareDiff, matrix


def getStudyRate(step):
    if step < 100:
        return 0.0001 + 0.00001*step
    else:
        return 0.1/step


def printShit(x):
    print('------------------------------')
    print(x)


def test1():
    numpy.random.seed(103)
    inputLen = 1
    # inputs = generateInputs(inputLen)
    inputs = numpy.random.rand(8, inputLen)
    print(inputs)
    model = getModelLayer(inputLen)
    model.input(inputs)
    outputs = model.outputs
    # avgIn, squareDiffIn, inputs = toOne(inputs)
    # avgOut, squareDiffOut, outputs = toOne(outputs)
    multiplyLine1 = selfMultiplyLayer(inputLen=len(inputs[0]), studyRate=0.001)
    printShit(multiplyLine1.matrix)
    print('----------------------------------------')
    oldCost = 5000
    cost = 0
    steps = 0
    count = 0
    for i in range(1000):
        # if i > 60:
        multiplyLine1.input(inputs)
        # multiplyLine1.printSelf()
        cost = getCost(multiplyLine1.outputs, outputs)
        # printShit(cost)

        if oldCost < cost:
            count += 1
            if count > 2 and multiplyLine1.studyRate > 0.0001:
                print('******************************************%d, %f, %f' % (i, multiplyLine1.studyRate, cost))
                multiplyLine1.studyRate = multiplyLine1.studyRate * 0.5
                count = 0
        else:
            if multiplyLine1.studyRate < 0.001:
                multiplyLine1.studyRate = multiplyLine1.studyRate + 0.00001
        oldCost = cost
        d_loss_o = get_d_loss_o(multiplyLine1.outputs, outputs)
        d_loss_o = multiplyLine1.backpropagation(d_loss_o)
        # print(d_loss_o)
        # print('----------------------------------------')
        steps = i
        if getCost(multiplyLine1.outputs, outputs) < 1e-6 or multiplyLine1.studyRate < 1e-6:
            break
    print('---------------------------------------------------------')
    print((multiplyLine1.outputs - outputs))
    print('---------------------------------------------------------')
    multiplyLine1.printSelf()
    print('-------------------------------')
    model.printSelf()
    printShit('steps = %d, cost = %f' % (steps, cost))
    # print(avgIn, squareDiffIn)
    # print(avgOut, squareDiffOut)


if __name__ == '__main__':
    test1()
