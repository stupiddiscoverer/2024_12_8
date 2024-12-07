import threading
from datetime import datetime

import numpy
import torch
import time
import os
import numpy as np
import random
from torch import nn, optim
from torch.nn import functional as F
from statistics import mean
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.deterministic = True


setup_seed(101)
inputLen = 1
hiddenLen = 4
batchSize = 128
testBatch = 1
trainBatch = 1024

xrange = 30


class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()
        self.l11 = nn.Linear(in_features=inputLen, out_features=hiddenLen)
        self.l12 = nn.Linear(in_features=inputLen, out_features=hiddenLen)
        self.l13 = nn.Linear(in_features=inputLen, out_features=hiddenLen)
        # self.norm1 = nn.BatchNorm1d()
        # self.l3 = nn.Linear(in_features=hiddenLen, out_features=hiddenLen)
        self.l31 = nn.Linear(in_features=hiddenLen, out_features=inputLen)
        self.l32 = nn.Linear(in_features=hiddenLen, out_features=inputLen)
        self.l33 = nn.Linear(in_features=hiddenLen, out_features=inputLen)
        # self.l4 = nn.Linear(in_features=hiddenLen, out_features=inputLen)

    def forward(self, x):
        out = F.leaky_relu(input=self.l11(x) + self.l12(x)*self.l13(x))
        out = F.leaky_relu(self.l31(out) + self.l32(out)*self.l33(out))
        return out


def loadNet():
    nett = Square()
    # device = torch.device('cuda', 0)
    # nett.to(device=device)
    nett.load_state_dict(torch.load('../torchReserve/square.pth'))
    # print(nett.l1.weight)
    plt.plot(nett.l11.weight.detach().numpy().reshape(hiddenLen), torch.linspace(0, hiddenLen, steps=hiddenLen), 'b.')
    plt.plot(nett.l31.weight.detach().numpy()[0].reshape(hiddenLen), torch.linspace(0, hiddenLen, steps=hiddenLen), 'g.')
    plt.show()
    # time.sleep(5)
    plt.close()
    x = torch.linspace(-xrange*1.5, xrange*1.5, 100).resize(100, 1)
    x = torch.div(x, xrange)
    x = torch.mul(x, 2)
    x = x.cpu()
    y = nett(x)

    print(nett.l11.weight.shape)
    x1 = x.detach().numpy().reshape(100)
    x1 = x1*xrange/2
    # print(x1)
    y1 = y.detach().numpy().reshape(100)
    print(y1)
    y1 = y1*xrange**2/4
    # print(y1)

    plt.plot(x1, y1, 'b.')
    plt.plot(x1, 100 - y1, 'r-')
    plt.show()


def startTrain():
    print(torch.cuda.current_device())
    device = torch.device('cuda', 0)
    # cmd: nvidia-smi   查看gpu利用率
    print(device)
    x = torch.rand(size=(trainBatch, batchSize, inputLen), dtype=torch.float32, device=device) * xrange*2 - xrange
    x = torch.div(x, xrange)
    x = torch.mul(x, 2)
    y = torch.pow(x, 2)

    z = torch.rand(size=(testBatch, batchSize, inputLen), dtype=torch.float32, device=device) * xrange*2 - xrange
    z = torch.div(z, xrange)
    z = torch.mul(z, 2)
    zz = torch.pow(z, 2)

    net = Square()
    net.to(device=device)
    print(net.modules())
    print(net(x[0]))
    # net.l1.weight *= 10
    # net.l3.weight *= 10
    # net.load_state_dict(torch.load('torchReserve/square.pth'))
    # adam是指数加权平均和指数加权平方平均法，用于减小每次迭代的偏导数震荡
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=3,
        threshold=0.001,
        verbose=True)
    # net.train()
    for epoch in range(100):
        b = []
        s = time.time()
        for i in range(trainBatch):
            in_x = x[i, :, :]
            out_y = y[i, :, :]
            out_x = net(in_x)
            loss = F.l1_loss(input=out_x, target=out_y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            b.append(loss.item())

        net.eval()
        a = []
        for j in range(testBatch):
            in_z = z[j, :, :]
            out_z = net(in_z)
            test_loss = F.l1_loss(input=out_z, target=zz[j, :, :])
            # print(out_z)
            # print(zz[j])
            a.append(test_loss.item())
        schedule.step(mean(a))
        e = time.time()
        print('epoch: ', epoch, 'test loss: ', mean(a), 'train loss: ', mean(b), 'time: ', e - s)

        if epoch % 10 == 9 or optimizer.param_groups[0]['lr'] <= 1e-5:
            torch.save(
                net.state_dict(),
                os.path.join('../torchReserve/', 'square.pth'))
        l1_regularization(net, 1/hiddenLen)
        # l2_regularization(net, 1/hiddenLen)


def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))


def l2_regularization(model, l2_alpha):
    for module in model.modules():
        if type(module) is nn.Conv2d:
            module.weight.grad.data.add_(l2_alpha * module.weight.data)


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    # startTrain()
    loadNet()

