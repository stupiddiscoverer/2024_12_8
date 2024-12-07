import numpy
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


# x = tf.Variable(3.0)
#
# with tf.GradientTape() as tape:
#     y = x**2
#     # dy = 2x * dx
# dy_dx = tape.gradient(y, x)
# print(dy_dx.numpy())
# print(numpy.array([[1, 2, 3], [4, 5, 6]]).astype(numpy.float32).T)
# w = tf.Variable(numpy.array([[1, 2, 3], [4, 5, 6]]).astype(numpy.float32).T, name='w')
# b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
# x = tf.Variable([[1., 2., 3.]])
#
# with tf.GradientTape(persistent=True) as tape:
#   y = x @ w + b
#   loss = tf.reduce_mean(y**2)
#
# [dl_dw, dl_db] = tape.gradient(y, [w, b])
# print(w.shape)
# print(dl_dw.shape)
# print(dl_dw)
# print(dl_db)
#
# my_vars = {
#     'w': w,
#     'b': b
# }
#
# grad = tape.gradient(loss, my_vars)
# print(grad['b'])
# (dl_dw, dl_db) = tape.gradient(loss, (w, b))
# print(dl_dw)
# print(dl_db)


layer = tf.keras.layers.Dense(3, activation='relu')
x = tf.constant([[1., 2., 3.]])

with tf.GradientTape() as tape:
  # Forward pass
  y1 = layer(x)
  print(y1)
  y = y1 * x
  print(y)
  loss = tf.reduce_mean(y**2)

# print(loss)
# Calculate gradients with respect to every trainable variable
grad = tape.gradient(loss, layer.trainable_variables)
for var, g in zip(layer.trainable_variables, grad):
  print(f'{var.name}, shape: {g.shape}')