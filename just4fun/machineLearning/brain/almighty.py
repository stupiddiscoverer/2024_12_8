# 一个自动选择路径的网络，网络本身大小固定，但是数据流经哪个区域，循环几次由数据本身决定
# 堆栈空间，堆空间存储持久性知识，栈空间存放临时推导的可信任知识，可以用于更新堆中矛盾知识，或直接输出结果，就像利用公式和步骤计算应用题答案
# 编程语言可以利用简单的逻辑运算和循环或跳跃执行实现任意可书写的函数

import os.path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 使用 PyTorch 2.0 的 torch.compile() 加速
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 保持尺寸不变
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(128 * 1 * 1, 10)  # 假设输入是 MNIST (28x28)，经过两次池化后为 7x7

    def forward(self, x):
        x = F.relu(self.conv1(x))       # [batch, 32, 28, 28]
        x = F.max_pool2d(x, 2)          # [batch, 32, 14, 14]
        x = F.relu(self.conv2(x))       # [batch, 64, 14, 14]
        x = F.max_pool2d(x, 2)          # [batch, 64, 7, 7]
        x = F.relu(self.conv3(x))       # [batch, 64, 7, 7]
        # print(x.shape)
        x = F.max_pool2d(x, 2)          # [batch, 64, 3, 3]
        # print(x.shape)
        x = F.relu(self.conv4(x))       # [batch, 64, 1, 1]
        x = torch.flatten(x, 1)          # [batch, 64*7*7 = 3136]
        return F.softmax(self.fc(x), dim=1)


def train():
    # 初始化并编译模型（PyTorch 2.0+ 特性）
    model = CNN().cuda()
    # model = torch.compile(model)  # 使用自动混合精度和优化

    # 数据加载
    train_data = datasets.MNIST('../torchTest/data', transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)


    print(f"训练集图片数量: {len(train_data)}")  # 输出: 60000
    # 训练循环（支持自动混合精度）
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.cuda.amp.GradScaler()  # FP16 加速

    for epoch in range(5):
        test(model)
        for x, y in train_loader:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x.cuda()), y.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # 保存整个模型（结构和参数）
    torch.save(model.state_dict(), 'Cnn_mnist.pth')

    # 如果需要保存模型结构+参数（完整恢复）
    torch.save(model, 'Cnn_mnist_full.pth')


def print_acc(model=None, test_loader=None):
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            total_correct += (torch.max(outputs, 1)[1] == y).sum().item()
            total_samples += y.size(0)
    accuracy = 100 * total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.2f}%')


def test(model=None):
    # 方法1：仅加载权重（需先实例化模型结构）
    # model = CNN()  # 需与训练时相同的模型类
    # model.load_state_dict(torch.load('model_weights.pth'))
    # model.eval()  # 切换到评估模式
    # 方法2：直接加载完整模型
    if model is None:
        model = torch.load('Cnn_mnist_full.pth')
        model.eval()
    # 加载测试数据
    test_data = datasets.MNIST(root='../torchTest/data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data, batch_size=10, shuffle=True)  # 随机取10张
    print(f"测试集图片数量: {len(test_data)}")  # 输出: 10000
    print_acc(model, test_loader)

    # 获取一个batch的数据
    images, labels = next(iter(test_loader))
    images, labels = images.cuda(), labels.cuda()  # 如果用GPU

    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 获取预测类别

    # 显示图片和预测结果
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Pred: {predicted[i].item()}\nTrue: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')  # 保存结果图片
    plt.show()


if __name__ == '__main__':
    print('hello')
    if os.path.exists('Cnn_mnist_full.pth'):
        test()
    else:
        train()
        test()