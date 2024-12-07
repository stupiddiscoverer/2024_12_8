import os
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchsummary import summary


def selfMul(tensor, dim=1):
    x = tensor.clone()
    len = x.shape[dim] // 2
    selfMul = tensor.narrow(dim, 0, len) * tensor.narrow(dim, len, len)
    x.narrow(dim, 0, len).copy_(selfMul)
    return x


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, groups=2)
        self.conv3 = nn.Conv2d(32, 64, 3, groups=2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 11)

    def forward(self, x):
        x = self.conv1(x)   # 26*26
        x = selfMul(x)
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)   # 11*11
        x = selfMul(x)
        x = nn.MaxPool2d(kernel_size=(2, 2))(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)  # 11*11
        x = selfMul(x)
        x = nn.MaxPool2d(kernel_size=(3, 3))(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = selfMul(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.1)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class linearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=3)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 11)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        # print(x.shape)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=-1)
        return x


class ResNet18ForMNIST(nn.Module):
    def __init__(self):
        super(ResNet18ForMNIST, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 使用预训练的ResNet18
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入层
        self.resnet.conv1.requires_grad_(False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 11)  # 假设有10个分类
        )
        # self.resnet.conv1.requires_grad_(True)

    def forward(self, x):
        return self.resnet(x)


class ViT(nn.Module):
    def __init__(self, in_channels=1, num_classes=11, dim=64, depth=4, heads=4, mlp_dim=64, dropout=0.1,
                 emb_dropout=0.1):
        super(ViT, self).__init__()

        self.patch_size = 4
        self.num_patches = (32 // self.patch_size) ** 2
        self.patch_dim = in_channels * self.patch_size ** 2
        self.dim = dim

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Transformer(d_model=dim, nhead=heads, num_encoder_layers=depth,
                                          dim_feedforward=mlp_dim, dropout=dropout)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(x.shape[0], -1, self.patch_dim)
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, x)

        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        return x


def trainDiscriminator():
    # loss 稳定在 0.001
    criterion = nn.MSELoss()
    # nn.CrossEntropyLoss()
    oldTime = time.time()
    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            oneHot = nn.functional.one_hot(labels, num_classes=11).float()
            optimizerDis.zero_grad()
            outputs = disNet(inputs)
            loss = criterion(outputs, oneHot)
            loss.backward()
            optimizerDis.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        # 保存模型
        torch.save(disNet.state_dict(), f'mnist_net_{modelName}.pth')
        newTime = time.time()
        print(f'saved, using {newTime - oldTime}s')
        oldTime = newTime
    print('Finished Training')


def showErrorImgs():
    disNet.eval()
    dataiter = iter(testloader)
    inputs, labels = next(dataiter)
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = disNet(inputs)
    _, predicted = torch.max(outputs, 1)
    errors = (predicted != labels).nonzero(as_tuple=True)[0]
    print(errors)
    print('errors = ', errors.shape[0])

    picNum = errors.shape[0]
    if picNum > 10:
        picNum = 10
    # 随机选择10张错误预测的图片
    random_errors = random.sample(list(errors.cpu().numpy()), picNum)

    # 显示图片和标签
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(picNum):
        idx = random_errors[i]
        image = inputs[idx].cpu().numpy().squeeze()
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()


if __name__ == '__main__':
    # modelName = "linearNet"
    # modelName = "Cnn"
    # print(time.time() / 60 / 60 / 24 / 365)
    # quit()
    modelName = 'ResNet18ForMNIST'
    inputShape = (1, 28, 28)
    function = globals().get(modelName)
    disNet = function()
    disNet.cuda()
    parameters = disNet.parameters()
    if modelName == 'ResNet18ForMNIST':
        inputShape = (1, 224, 224)
        parameters = []
        for name, param in disNet.named_parameters():
            if param.requires_grad:
                parameters.append(param)
    summary(disNet, inputShape)
    optimizerDis = optim.Adam(parameters, lr=0.001)

    batchSize = 128
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if modelName == "ResNet18ForMNIST":
        transform = transforms.Compose([
            transforms.Resize(224),  # ResNet需要224x224的输入
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=True)

    # 实例化神经网络模型
    # disNet = ViT()
    if os.path.exists(f'mnist_net_{modelName}.pth'):
        disNet.load_state_dict(torch.load(f'mnist_net_{modelName}.pth'))
    trainDiscriminator()
    showErrorImgs()
    
    