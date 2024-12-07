import numpy as np
import tensorflow as tf

board = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]

# board = np.array(board)
print(tf.__version__)


def getDotModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, use_bias=True, input_shape=(2,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

    model.add(tf.keras.layers.Reshape((2, 2, 1)))

    model.add(tf.keras.layers.Conv2DTranspose(2, (5, 5), strides=(2, 2), padding='valid', use_bias=False))
    assert model.output_shape == (None, 7, 7, 2)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))

    model.add(tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='valid', use_bias=False))
    assert model.output_shape == (None, 9, 9, 1)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Reshape((9, 9)))
    return model


def get_dot_board(x, y):
    board1 = np.array(board, dtype='float32')
    board1[x][y] = 1.0
    board1 = board1.reshape((1, 9, 9))
    return tf.cast(board1, tf.float32)


optimizer = tf.keras.optimizers.Adam(1e-4)


loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def get_loss(real_images, generated_images):
    return loss_func(real_images, generated_images)


@tf.function
def train_step(coord, real_images):
    with tf.GradientTape() as dot_tape:
        generated_images = dotModel(coord, training=True)   # 中间变量必须有名字，不能用函数返回值作为输入，例如xx.toList()
        gen_loss = get_loss(real_images, generated_images)
        # print(gen_loss)
    gradients_of_generator = dot_tape.gradient(gen_loss, dotModel.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_generator, dotModel.trainable_variables))


dotModel = getDotModel()
print(dotModel.summary())
for k in range(200):
    print(k)
    for i in range(9):
        for j in range(9):
            train_step(tf.convert_to_tensor([[i, j]], dtype='float32'), get_dot_board(i, j))

generated_images = dotModel(tf.convert_to_tensor([[5, 4]], dtype='float32'))
print(generated_images)

