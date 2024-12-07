import tensorflow as tf
from tensorflow import keras

from machineLearning.selfMultiplyNet import generateTrainData


class DotProductLayer(keras.layers.Layer):
    def __init__(self):
        super(DotProductLayer, self).__init__()

    def call(self, inputs):
        x, y = inputs[0], inputs[1]
        return tf.multiply(x, y)


input1 = keras.layers.Input(shape=(1,))
x = keras.layers.Dense(1, activation="relu", name="dense_1")(input1)
dot_product = DotProductLayer()([input1, x])
model = keras.models.Model(inputs=[input1], outputs=dot_product)
model.compile(
    # optimizer=keras.optimizers.Adam(),  # Optimizer
    # optimizer='rmsprop',
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.mean_squared_error,
    # List of metrics to monitor
    metrics=[keras.metrics.MeanSquaredError()]
    # metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

x_train, y_train = generateTrainData(1)

history = model.fit(
    x_train,
    y_train,
    # y_train_one_hot,
    batch_size=4,
    epochs=400,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_train, y_train),
)

print('------------------')
print(history)
print('------------------')
print(model.history)
print('------------------')
print(model.layers)
