import torch
import torch.nn as nn


# 定义一个示例模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)
        nn.Transformer()

    def forward(self, x):
        return self.fc(x)


model = Net()

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = Net()
model.load_state_dict(torch.load('model.pth'))
print(model.state_dict())
model.eval()  # 设置为评估模式
