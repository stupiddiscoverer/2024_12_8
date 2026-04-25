import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. 用于图像读取和预处理
def image_loader(image_path, imsize):
    loader = transforms.Compose([
        transforms.CenterCrop(min(Image.open(image_path).size)),  # 先裁成正方形（以短边为准）
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

# 2. 显示图片
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# 3. 内容损失
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# 4. 风格损失
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    # print('a * b * c * d = ', a * b * c * d)
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# 5. 加载图片
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
content_img = image_loader("C:/Users/张三/Pictures/浏览器/fuck.jpg", imsize).to(device)
style_img = image_loader('C:/Users/张三/Pictures/浏览器/starSky.jpg', imsize).to(device)

# 6. 加载预训练的VGG网络
cnn = models.vgg19(weights='IMAGENET1K_V1').features.to(device).eval()
# VGG19的预处理均值和方差
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 7. 定义归一化模块
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# 8. 损失层插入位置
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# 9. 构建模型
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                              style_img, content_img,
                              content_layers, style_layers):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses

model, style_losses, content_losses = get_style_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std,
    style_img, content_img, content_layers, style_layers)

# 10. 输入初始化
input_img = content_img.clone()

# 11. 定义优化器和风格迁移步骤
def run_style_transfer(model, style_losses, content_losses, input_img, num_steps=300,
                      style_weight=1e6, content_weight=1):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
            return loss
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img


# 12. 启动风格迁移
output = run_style_transfer(model, style_losses, content_losses, input_img)
imshow(output, 'Output Image')

