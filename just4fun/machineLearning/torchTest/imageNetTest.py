import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchsummary import summary

# 检查是否可以使用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据增强和预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 加载 CIFAR-100 数据集
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 获取迭代器
# data_iter = iter(train_loader)
#
# # 获取第一个 batch 的数据和标签
# inputs, labels = next(data_iter)
# print("DataLoader 的长度（batch 数量）为:", len(train_loader))
#
# # 打印第一个 batch 的数据大小和标签
# print("第一个 batch 的数据大小:", inputs.size())
# print("第一个 batch 的标签大小:", labels.size())
# quit()

# 加载预训练的 ResNet-18 模型（适用于较小数据集）
model = models.resnet18(pretrained=True)

# 修改最后的全连接层以匹配 CIFAR-100 的类别数（100 类）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)
model.load_state_dict(torch.load('resnet18_cifar100.pth'))

model = model.to(device)
print(summary(model, input_size=(3, 32, 32)))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和验证函数
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects.double() / len(test_dataset)

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# 训练模型
model = train_model(model, criterion, optimizer, train_loader, test_loader)

# 保存模型
torch.save(model.state_dict(), 'resnet18_cifar100.pth')
