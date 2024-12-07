import os

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def  generateImagelist():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = train_images / 127.5 - 1  # 将图像范围归一化为[-1, 1]
    imageList = []
    for i in range(10):
        imageList.append([])

    for i in range(60000):
        imageList[train_labels[i]].append(train_images[i])

    for i in range(10):
        if len(imageList[i]) > 6000:
            imageList[i] = imageList[i][0:6000]
        else:
            need = 6000 - len(imageList[i])
            for j in range(need):
                imageList[i].append(imageList[i][j])
    imageList = numpy.array(imageList)
    return imageList


imageList = generateImagelist()
print(imageList.shape)
BATCH_SIZE = 16


def getTrainDataSet():
    arr = []
    for i in range(10):
        arr.append(tf.data.Dataset.from_tensor_slices(imageList[i]).shuffle(int(imageList[i].size / 28 / 28)).batch(
            BATCH_SIZE))

    return tf.convert_to_tensor(arr)


def convTransLayer(layerNum, strides, inputs):
    outputs = layers.Conv2DTranspose(layerNum, (7, 7), strides=strides, use_bias=False, padding='same')(inputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.LeakyReLU()(outputs)
    return outputs


def make_generator_model():  # 反向卷积
    inputs = keras.Input(shape=(noise_dim,), name="digits")
    x11 = layers.Dense(7 * 7 * 1, use_bias=False)(inputs)
    # x12 = layers.Dense(7 * 7 * 2, use_bias=False)(inputs)
    # x13 = layers.Dense(7 * 7 * 2, use_bias=False)(inputs)
    # x14 = layers.LeakyReLU()(x11)
    x15 = layers.Reshape((7, 7, 1))(x11)

    x21 = convTransLayer(32, (1, 1), x15)
    x31 = convTransLayer(64, (2, 2), x21)
    x51 = convTransLayer(32, (2, 2), x31)
    # x51 = convTransLayer(32, (1, 1), x41)

    x61 = layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x51)
    x62 = layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x51)
    x63 = layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x51)

    # outputs = layers.LeakyReLU()(x61 + x63 * x62)
    outputs = layers.LeakyReLU()(x61)
    model = keras.Model(inputs=inputs, outputs=outputs)
    # model = tf.keras.Sequential()
    # # model1 = tf.keras.Model()
    # model.add(layers.Dense(7 * 7 * 32, use_bias=False, input_shape=(noise_dim,)))
    # # 输入100，输出7*7*64？这不是少数数据产生多数数据吗？
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Reshape((7, 7, 32)))
    # assert model.output_shape == (None, 7, 7, 32)
    #
    # model.add(layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 7, 7, 64)
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 16)
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(
    #     layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 28, 28, 1)
    return model


def convLayer(layerNum, strides, inputs):
    x1 = layers.Conv2D(layerNum, (7, 7), strides=strides, use_bias=False, padding='same')(inputs)
    # x2 = layers.BatchNormalization()(x1)
    x3 = layers.LeakyReLU()(x1)
    return x3


def make_discriminator_model():
    inputs = keras.Input(shape=(28, 28, 1), name="digits")
    # print(type(inputs))
    x11 = convLayer(32, (1, 1), inputs)
    x21 = convLayer(64, (2, 2), x11)
    x41 = convLayer(32, (2, 2), x21)
    # x41 = convLayer(32, (1, 1), x31)

    x51 = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same')(x41)
    x52 = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same')(x41)
    x53 = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same')(x41)
    x54 = layers.BatchNormalization()(x51 + x52 * x53)
    x55 = layers.LeakyReLU()(layers.Flatten()(x51))

    x61 = layers.Dense(10, use_bias=False)(x55)
    x62 = layers.LeakyReLU()(x61)

    outputs = layers.LeakyReLU()(layers.Dense(1, use_bias=False)(x62))
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
lossFunc = tf.keras.losses.Loss()


# def getTargets(n):
#     arr = numpy.zeros((BATCH_SIZE, noise_dim))
#     for i in range(BATCH_SIZE):
#         arr[i][n] = 1.0
#     return tf.convert_to_tensor(arr)


def discriminator_loss(real_output, targets):
    real_loss = lossFunc(targets, real_output)  # real_output为1最好？
    return real_loss


def generator_loss(fake_output, targets):
    return lossFunc(targets, fake_output)  # fake_output为1最好，意思是产生的很像真实图像


generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)


@tf.function
def train_step():
    for j in range(10):
        for i in range(len(inputBatches[0])):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                generated_images = generator(inputBatches[j][i][0], training=True)
                real_output = discriminator(inputBatches[j][i][1], training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output, inputBatches[j][i][0])
                disc_loss = discriminator_loss(real_output, inputBatches[j][i][0])
                # print(gen_loss, disc_loss)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def getRandomVector(num):
    arr = numpy.random.rand(10, num, 10) / 8
    for i in range(10):
        for j in range(num):
            arr[i][j][i] = 1.0
    return tf.convert_to_tensor(arr)


def generateNoiseInput(num):
    noise_input = numpy.random.rand(BATCH_SIZE, noise_dim) / 8
    for i in range(BATCH_SIZE):
        noise_input[i][num] = 1.0
    return tf.convert_to_tensor(noise_input)


def generateShowInput():
    showInputs = numpy.random.rand(10, noise_dim) / 8
    for i in range(10):
        showInputs[i][i] = 1.0
    return showInputs


def showImages(images):
    print(images.shape)
    plt.figure(figsize=(3, 4))
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.title(label=str(i))
        plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()
    return


def loadModel():
    generatorModel = keras.models.load_model('ganTestSave/generator.h5')
    generated_images = []
    for i in range(10):
        generated_images.append(generator(random_vector_for_generation[i], training=False)[0])
    showImages(tf.convert_to_tensor(generated_images))


if __name__ == '__main__':
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    EPOCHS = 5
    noise_dim = 10
    num_examples_to_generate = 16

    # random_vector_for_generation = tf.random.normal([num_examples_to_generate, noise_dim])
    random_vector_for_generation = getRandomVector(num_examples_to_generate)
    generator = make_generator_model()
    print(generator.summary())
    discriminator = make_discriminator_model()
    print(discriminator.summary())
    print('discriminator ' + str(discriminator.output_shape))
    train_dataset = getTrainDataSet()
    inputBatches = []
    for i in range(10):
        inputBatches.append([])
        print(f'\r%d' % i, end='')
        noise_input = generateNoiseInput(i)  # (64, 28, 28, 1)
        inputBatches[i].append(noise_input)
        print('\r', end='')
    inputBatches = tf.convert_to_tensor(inputBatches)
    for epoch in range(EPOCHS):
        print("Epoch:", epoch)
        generated_images = generator(generateShowInput(), training=False)
        showImages(generated_images)
        train_step()

    generated_images = generator(generateShowInput(), training=False)
    showImages(generated_images)
    # generator.save('ganTestSave/generator.h5')
    # discriminator.save('ganTestSave/discriminator.h5')