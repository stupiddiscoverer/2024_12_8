import numpy

from machineLearning.neuralNetwork import NeuralNet


# 3*3的棋盘，2位选手先连成一条线的赢

def checkWinOrDogFall(board):
    # 一共8种连线方式
    if board[0][0] == board[0][1] == board[0][2]:
        return board[0][0]
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][0] == board[1][0] == board[2][0]:
        return board[0][0]
    if board[0][1] == board[1][1] == board[2][1]:
        return board[0][1]
    if board[0][2] == board[1][2] == board[2][2]:
        return board[0][2]
    if board[1][0] == board[1][1] == board[1][2]:
        return board[1][0]
    if board[2][0] == board[2][1] == board[2][2]:
        return board[2][0]
    if board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    return -1


if __name__ == '__main__':
    board = numpy.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    print(board)
    board = -numpy.ones((3, 3))
    print(board)
    nnet = NeuralNet()
