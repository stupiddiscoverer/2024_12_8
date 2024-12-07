import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# CIFAR-100 数据集的类别
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载 CIFAR-100 数据集
cifar100_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 随机选择10张图片
random_indices = np.random.choice(len(cifar100_dataset), 10, replace=False)


def show10Img(imgs, labels):
    # 显示图片及其分类
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Random CIFAR-100 Images and their Labels')
    for i in range(len(imgs)):
        img,label = imgs[i], labels[i]
        img = img.permute(1, 2, 0).numpy()  # 调整维度以适应 matplotlib 的显示
        img = Image.fromarray((img * 255).astype(np.uint8))  # 转换为 PIL 图像
        # img = img.resize((128, 128), Image.NEAREST)  # 放大图像
        ax = axs[i // 5, i % 5]
        ax.imshow(img)
        ax.set_title(cifar100_classes[label])
        ax.axis('off')

    plt.show()

imgs = []
labels = []
for i, idx in enumerate(random_indices):
    img, label = cifar100_dataset[idx]
    imgs.append(img)
    labels.append(label)

show10Img(imgs, labels)
