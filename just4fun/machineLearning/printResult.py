import numpy
import numpy as np

def printResult(matrix1, matrix2, matrix3, bias):
    # i*(m1*m2+m3)
    shape = matrix1.shape
    length = shape[0]
    output = np.zeros_like(matrix1.T)
    output2 = np.zeros((shape[1], shape[0], shape[0]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            output[j][i] = matrix3[i][j]
            for k in range(shape[0]):
                output2[j][i][k] = matrix1[i][j] * matrix2[k][j]
    print(output)
    for i in range(shape[1]):
        for j in range(shape[0]):
            print(str(output2[i][j][j]) + chr(ord('a') + j) + "^2", end=' + ')
            for k in range(j+1, shape[0]):
                # output2[i][j][k] += output2[i][k][j]
                # output2[i][k][j] = 0
                print(str(output2[i][j][k] + output2[i][k][j]) + chr(ord('a') + j) + chr(ord('a') + k), end=" + ")
            print(str(output[i][j]) + chr(ord('a') + j), end=' + ')
        print(bias[i])

    print(output2)


m1 = numpy.array([[-0.83075334, 0.04422317], [-0.03222435, -0.63467021]])
m2 = numpy.array( [[-0.84116766, -0.07846039], [ 0.03166327, -1.10097629]])
m3 = numpy.array( [[-2.00030313e+00, -1.05554943e-03], [ 5.71875456e-04, -1.99964715e+00]])
b = numpy.array([1.00376798, 1.00648054])
# 0.7 * inputs ** 2 - 2 * inputs + 1
#  [ 0.35644658 -0.46956746]  [ 0.37604475  2.09348043]  [ 0.37692872  2.09854876]
printResult(m1, m2, m3, b)