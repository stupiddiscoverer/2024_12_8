import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 定义输入图像的大小
input_shape = (256, 256, 3)

# 定义输入层
inputs = Input(shape=input_shape)

# 加载预训练的VGG19模型
vgg = VGG19(include_top=False, weights='imagenet', input_tensor=inputs)

# 获取VGG19模型的特征图
outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

# 指定用于提取内容特征的层
content_layer = 'block5_conv2'

# 指定用于提取风格特征的层
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# 定义内容特征提取模型
content_model = Model(inputs=vgg.input, outputs=outputs_dict[content_layer])

# 定义风格特征提取模型
style_models = [Model(inputs=vgg.input, outputs=outputs_dict[layer]) for layer in style_layers]


# 定义特征提取函数
def get_features(model, preprocessed_input):
    """获取特征"""
    outputs = model(preprocessed_input)
    features = tf.squeeze(outputs, axis=0)
    return features


# 定义Gram矩阵函数
def gram_matrix(input_tensor):
    """计算Gram矩阵"""
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


# 定义内容损失函数
def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


# 定义风格损失函数
def style_loss(base_style, gram_target):
    print(base_style)
    print(gram_target)
    """计算风格损失"""
    gram_style = gram_matrix(base_style)
    print(gram_style)
    loss1 = tf.reduce_mean(tf.square(gram_style - gram_target))
    print(loss1)
    return loss1


# 定义总变差损失函数
def total_variation_loss(x):
    """计算总变差损失"""
    a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return tf.reduce_mean(tf.pow(a + b, 1.25))


# 定义风格迁移模型
def style_transfer(content_image, style_image, num_iterations=1000, content_weight=1e3, style_weight=1e-2, tv_weight=1e-4):
    """风格迁移"""
    # 预处理输入图像
    content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image)

    # 提取内容特征
    content_features = get_features(content_model, content_image)
    style_features = [get_features(model, style_image) for model in style_models]

    # 计算风格特征的Gram矩阵
    style_gram_targets = [gram_matrix(feature) for feature in style_features]

    # 初始化生成的图像
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    # 定义优化器
    optimizer = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # 记录损失值
    content_loss_history = []
    style_loss_history = []
    tv_loss_history = []
    total_loss_history = []

    # 迭代优化生成的图像
    for i in range(num_iterations):
        # 计算生成图像的内容特征
        generated_features = get_features(content_model, generated_image)

        # 计算生成图像的风格特征的Gram矩阵
        generated_gram_targets = [gram_matrix(feature) for feature in generated_features]

        # 计算内容损失
        content_loss_value = content_loss(content_features, generated_features)

        # 计算风格损失
        style_loss_value = 0
        for j in range(len(style_features)):
            style_loss_value += style_loss(generated_features[j], style_gram_targets[j])
        style_loss_value *= style_weight / len(style_features)

        # 计算总变差损失
        tv_loss_value = tv_weight * total_variation_loss(generated_image)

        # 计算总损失
        total_loss_value = content_weight * content_loss_value + style_loss_value + tv_loss_value

        # 更新生成的图像
        grads = tf.gradients(total_loss_value, generated_image)[0]
        optimizer.apply_gradients([(grads, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0, 255))

        # 记录损失值
        content_loss_history.append(content_loss_value.numpy())
        style_loss_history.append(style_loss_value.numpy())
        tv_loss_history.append(tv_loss_value.numpy())
        total_loss_history.append(total_loss_value.numpy())

        # 输出损失值
        if i % 100 == 0:
            print('Iteration {}, Total loss: {:.4e}, Content loss: {:.4e}, Style loss: {:.4e}, TV loss: {:.4e}'
                  .format(i, total_loss_value.numpy(), content_loss_value.numpy(), style_loss_value.numpy(), tv_loss_value.numpy()))

    # 返回生成的图像和损失值历史记录
    return generated_image.numpy(), content_loss_history, style_loss_history, tv_loss_history, total_loss_history


# 加载输入图像
content_image = np.array(Image.open('C:\\Users\\fly\\Pictures\\情头\\1585102925475458.jpg').resize((256, 256)))
style_image = np.array(Image.open('C:\\Users\\fly\\Pictures\\Nitro\\Nitro_Wallpaper_5000x2813.jpg').resize((256, 256)))
plt.imshow(content_image)
plt.show()
plt.imshow(style_image)
plt.show()
content_image = content_image.reshape((1, 256, 256, 3))
style_image = style_image.reshape((1, 256, 256, 3))
# 运行风格迁移算法
generated_image, content_loss_history, style_loss_history, tv_loss_history, total_loss_history = style_transfer(content_image, style_image)

# 显示生成的图像
plt.imshow(generated_image.astype(np.uint8))
plt.show()
