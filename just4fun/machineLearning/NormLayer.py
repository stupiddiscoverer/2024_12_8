import numpy


class NormLayer:
    def __init__(self):
        self.rate = 1.0
        self.b = 0.0

    def forward(self, inputs):
        sum = numpy.sum(inputs)
        num = 1
        for s in inputs.shape:
            num *= s
        avg = sum / num
        variance = 0.0
        for s in numpy.reshape(inputs, num):
            variance += (s - avg) ** 2
        variance /= num
        variance = variance**0.5
        self.rate = 1 / (variance + 1e-6)
        self.b = -avg
        return (inputs + self.b) * self.rate

    def backward(self, gradients):
        return gradients * self.rate


def test():
    nl = NormLayer()
    a = numpy.array([[1,2,3,4],[5,6,7,8]])
    b = nl.forward(a)
    print(b)
    print(nl.backward(b))