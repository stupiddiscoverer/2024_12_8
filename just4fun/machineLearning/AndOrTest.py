import numpy


def andAnd(a, b):
    c = a.copy()
    for i in range(len(a)):
        c[i] = a[i] & b[i]
    return c


def print4B(x):
    y = bin(x)[2:]
    for i in range(4 - len(y)):
        y = '0' + y
    print(y)


def test():
    input1 = numpy.array([0b1011, 0b0111])
    hidden = numpy.array([0b1011, 0b1100])

    result = andAnd(input1, hidden)
    for i in range(len(hidden)):
        print4B(result[i])
    print4B(result[0] & result[1])


if __name__ == '__main__':
    test()
# output = [0b0001]
