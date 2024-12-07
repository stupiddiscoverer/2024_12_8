import os.path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


# 定义Basic Block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


# 定义ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def trainNet():
    # 加载数据

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    # 训练网络
    for epoch in range(5):  # 进行5个epoch的训练
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个mini-batches输出一次损失
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        torch.save(resnet.state_dict(), 'resnet18_cifar10.pth')

    print('Finished Training')


def showErrorImgs():
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
    resnet.eval()

    # 获取测试数据和标签
    dataiter = iter(testloader)
    inputs, labels = next(dataiter)
    # for i, data in enumerate(testloader, 0):
    #     inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    # 进行预测
    outputs = resnet(inputs)
    _, predicted = torch.max(outputs, 1)

    # 找出错误的预测
    errors = (predicted != labels).nonzero(as_tuple=True)[0]
    print(errors)
    print('error rate is ', errors.shape[0] / 256)

    # 随机选择10张错误预测的图片
    random_errors = random.sample(list(errors.cpu().numpy()), 10)

    # 显示图片和标签
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(10):
        idx = random_errors[i]
        image = inputs[idx].cpu().numpy().transpose((1, 2, 0))
        image = np.clip(image, 0, 1)
        true_label = classes[labels[idx].item()]
        pred_label = classes[predicted[idx].item()]
        axes[i].imshow(image)
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建ResNet18模型
    resnet = ResNet18()
    resnet.to(device)
    if os.path.exists("resnet18_cifar10.pth"):
        resnet.load_state_dict(torch.load("resnet18_cifar10.pth"))
    # trainNet()
    showErrorImgs()