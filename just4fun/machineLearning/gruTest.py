import numpy

from machineLearning import funcSet
from machineLearning.neuralNetwork import NeuralNet
from machineLearning.neuralNetwork import NeuralLine


class gru:
    def __init__(self, inputLen, outputLen):
        self.inputNet = NeuralNet(inputLen, [inputLen, inputLen], [funcSet.reLuSet, funcSet.reLuSet], [0.1, 0.1])
        self.hiddenNet = NeuralNet(inputLen, [inputLen, inputLen], [funcSet.reLuSet, funcSet.reLuSet], [0.1, 0.1])
        self.hiddenInput = numpy.ndarray((1, inputLen))
        self.input = numpy.ndarray((1, inputLen))
        self.

    def forward(self):
        self.hiddenInput = self.hiddenInput * (funcSet.sigmoidSet.func(self.inputNet.input(self.input) + self.inputNet))
        funcSet.tanhSet.func(self.hiddenInput + self.inputNet.netEnd.outputs)
