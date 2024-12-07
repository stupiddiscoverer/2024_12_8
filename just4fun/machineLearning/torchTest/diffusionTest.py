import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# 定义稳定扩散生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义生成器网络结构，这里只是一个简单的示例
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


# 初始化生成器模型
generator = Generator()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.001)

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练生成器模型
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataloader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1)

        # 生成随机噪声
        z = torch.randn(batch_size, 100)

        # 优化器梯度清零
        optimizer.zero_grad()

        # 使用生成器生成图片
        generated_img = generator(z)

        # 计算损失
        loss = criterion(generated_img, images)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

print('Finished Training')
