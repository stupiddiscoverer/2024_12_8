import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# 定义神经网络模型
class discriminatorNet(nn.Module):
    def __init__(self):
        super(discriminatorNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2, groups=16)
        self.conv3 = nn.Conv2d(64, 64, 4, groups=16)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 11)

    def forward(self, x):
        x = self.conv1(x)   # 26*26
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        # print(x.shape)
        x = self.conv2(x)   # 11*11
        x = nn.functional.relu(x)
        # print(x.shape)
        x = self.conv3(x)  # 11*11
        x = nn.functional.relu(x)
        # x = x.flatten()
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.fc2(x)
        # x = nn.functional.softmax(x, dim=1)
        return x


class ganNet(nn.Module):
    def __init__(self):
        super(ganNet, self).__init__()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 64*3*3)
        self.rconv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0)  # 7*7
        self.rconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)  # 13*13
        self.rconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0)   # 28*28
        self.conv3 = nn.Conv2d(32, 1, 3, stride=1, padding=1)   # 28*28

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.fc2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = x.view(-1, 64, 3, 3)
        x = self.rconv1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.rconv2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.rconv3(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.conv3(x)
        return x


batchSize = 64
# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)

# 实例化神经网络模型
disNet1 = discriminatorNet()
ganNet1 = ganNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()
# nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizerDis = optim.Adam(disNet1.parameters(), lr=0.001)
optimizerGan = optim.Adam(ganNet1.parameters(), lr=0.001)


def trainDiscriminator():
    # loss 稳定在 0.001
    # dicts = torch.load('mnist_net.pth')
    # 训练模型
    disNet1.cuda()
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            oneHot = nn.functional.one_hot(labels, num_classes=11).float()
            optimizerDis.zero_grad()
            outputs = disNet1(inputs)
            loss = criterion(outputs, oneHot)
            loss.backward()
            optimizerDis.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        # 保存模型
        torch.save(disNet1.state_dict(), 'mnist_net.pth')
    print('Finished Training')


def onlyGanTrain():
    # loss 稳定在 0.21 但是还算是图片，像是图片叠加的平均，有点模糊，
    # 如果数据集中的图片不那么居中，也不那么正，大小也不那么相似，那就没法训练了
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            if images.shape[0] != batchSize:
                break
            optimizerGan.zero_grad()
            labelHots = nn.functional.one_hot(labels, num_classes=11).float()
            randInputs = torch.rand((batchSize, 11)) / 10

            outputs = ganNet1(labelHots + randInputs)
            loss = criterion(outputs, images)
            loss.backward()
            optimizerGan.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # 保存模型
        torch.save(ganNet1.state_dict(), 'onlyGan.pth')
        testGan('onlyGan.pth', 'mnist_net.pth')
    print('Finished Training')
    return


def adversarialTrain():
    # 对抗训练某一个网络收敛过快会导致这个网络无法更新，无法学习，因为损失值很小，
    # 导致另一个网络不管怎么努力总是不是该网络的对手，无法学到有用，有指导作用的知识
    print(f"Using device: {device}")
    ganNet1.to(device)
    disNet1.to(device)
    for epoch in range(5):
        gan_loss = 0.0
        dis_loss = 0.0
        mosaic = nn.functional.one_hot(torch.ones(batchSize, dtype=torch.long) * 10, num_classes=11).float().to(device)
        # print(mosaic)
        count = 0
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            if images.shape[0] != batchSize:
                break
            images = images.to(device)
            labels = labels.to(device)
            labelHots = nn.functional.one_hot(labels, num_classes=11).float().to(device)
            optimizerGan.zero_grad()
            optimizerDis.zero_grad()

            disNet1.requires_grad_(False)
            randInputs = torch.rand((batchSize, 11)) / 10
            randInputs = randInputs.to(device)
            ganOutputs = ganNet1(labelHots + randInputs)
            ganOutputs1 = disNet1(ganOutputs)
            lossGan = criterion(ganOutputs1, labelHots)
            lossGan.backward()
            optimizerGan.step()
            gan_loss += lossGan.item()
            disNet1.requires_grad_(True)

            ganNet1.requires_grad_(False)
            ganOutputs = ganNet1(labelHots + randInputs).detach()
            ganOutputs1 = disNet1(ganOutputs)
            disOutput1 = disNet1(images)
            lossDis = criterion(ganOutputs1, (mosaic+labelHots)/2) + criterion(disOutput1, labelHots)
            # 先注重提高鉴别器对真实图片的鉴别能力，使得鉴别器更像人，再提高假图片的鉴别能力，这样就可以一步步提高生成器造假能力？
            # 实际效果并没有提高，生成器比重最佳就是0.2，0.1和0.3都不行，过低导致生成器进化过快，学不到东西，过高导致鉴别器进化过快
            if lossDis.item() >= lossGan.item() or count < 10:
                # 因为鉴别器学习能力强过生成器（因为学习目标简单），所以减缓鉴别器进化
                lossDis.backward()
                count += 1
            # lossDis = -criterion(ganOutputs1, labelHots) * 0.2 + criterion(disOutput1, labelHots)
            # 这个损失函数不如上面的好
            optimizerDis.step()
            dis_loss += lossDis.item()
            ganNet1.requires_grad_(True)
            if i % 100 == 99:
                print('[%d, %5d] gan_loss: %.5f  dis_loss: %.5f count=%d' %
                      (epoch + 1, i + 1, gan_loss / 100, dis_loss / 100, count))
                gan_loss = 0.0
                dis_loss = 0.0
                count = 0

        # 保存模型
        torch.save(ganNet1.state_dict(), 'gan_net.pth')
        torch.save(disNet1.state_dict(), 'dis_net.pth')
        testGan()
    print('Finished Training')
    return


def leftMov(imgs):
    # for x in range(imgs.shape[0])
    arr = imgs.numpy()
    # arr = numpy.random.rand(4,1,2,6)
    # print(arr)
    for k in range(arr.shape[0]):
        for i in range(arr.shape[3] - 1):
            for j in range(arr.shape[2]):
                arr[k][0][j][i] = arr[k][0][j][i+1]
    # print(arr)
    return torch.tensor(arr)


def testDiscriminator():
    net1 = discriminatorNet()
    net1.load_state_dict(torch.load('mnist_net.pth'))
    # plt.figure(figsize=(2, 2))
    for i, data in enumerate(trainloader, 0):
        print(i, len(data))
        inputs, labels = data
        # mixInputs = (inputs[0:4] + inputs[4:8])
        print(numpy.average(inputs))
        labels = labels[0:8]
        # mixOuts = net1(mixInputs)
        Outs = net1(inputs[0:8])
        # print(labels, mixOuts, Outs)

        leftMovInputs = leftMov(inputs[0:8])
        leftMovInputs = leftMov(leftMovInputs)
        leftMovOuts = net1(leftMovInputs)
        print(labels)
        for i in range(8):
            print(Outs[i])
            print(leftMovOuts[i])
            print('-----------------------------------------')

        # for j in range(4):
        #     plt.subplot(1, 4, j+1)
        #     plt.imshow(mixInputs[j][0], cmap='gray')
        #     plt.axis('off')
        # plt.show()
        break


def testGan(ganPth='gan_net.pth', disPth='dis_net.pth'):
    ganNet1.load_state_dict(torch.load(ganPth))
    disNet1.load_state_dict(torch.load(disPth))
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        disOut = disNet1(images[0])
        print(labels[0], disOut)
        ganOut = ganNet1(disOut)
        # print(ganOut.shape)
        print(disNet1(ganOut))
        ganOut = ganOut.to(torch.device("cpu"))
        plt.imshow(ganOut.detach().numpy()[0][0], cmap='gray')
        plt.show()
        break
    for i in range(100):
        oneHot = nn.functional.one_hot(torch.tensor(i % 10), num_classes=11).float().to(device)
        randInputs = torch.rand((1, 11)) / 10
        randInputs = randInputs.to(device)
        plt.subplot(10, 10, i + 1)
        ganOut = ganNet1(oneHot + randInputs).to(torch.device("cpu"))
        plt.imshow(ganOut.detach().numpy()[0][0], cmap='gray')
    plt.show()

    return


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(torch.rand((1, 11)) / 10)
    adversarialTrain()
    # testGan()
    # trainDiscriminator()
    # onlyGanTrain()
    # testGan('onlyGan.pth', 'mnist_net.pth')
    testGan()
    # testDiscriminator()
