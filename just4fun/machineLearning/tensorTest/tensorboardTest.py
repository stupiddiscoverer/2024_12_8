import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])
print(x @ tf.transpose(x))