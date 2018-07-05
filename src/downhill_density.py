import tensorflow as tf
import numpy as np

class Density():
    def __init__(self, n_hidden, width, depth):
        self.num_units = 1
        self.width = width
        self.depth = depth

        self.construct()

    def construct(self):
        layers = []
        layers.append(tf.keras.layers.Conv2D(self.width, 4, strides=(2, 2),
                                   padding='same',
                                   # input_shape=(28,28,1)
                                   ))
        layers.append(tf.keras.layers.Activation(tf.keras.activations.selu))
        for i in range(self.depth):
            layers.append(tf.keras.layers.Conv2D(self.width,
                                                 4,
                                                 strides=(2, 2),
                                                 padding='same'),)
            layers.append(tf.keras.layers.Activation(tf.keras.activations.selu))
        layers.append(tf.keras.layers.Conv2D(self.num_units,
                                1,
                                strides=(1, 1),
                                padding='same'))
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        layers.append(tf.keras.layers.Activation(tf.nn.sigmoid))
        self.cnn = tf.keras.Sequential(layers)

    def __call__(self, x):
        """
        Args:
            x (tf.tensor): the input
                shape is [None, width, height, channels],
                dtype is tf.float32
        """
        with tf.name_scope('classifier'):
            return self.cnn(x)

    def make_losses(self, x):
        with tf.name_scope('loss'):
            p = self.cnn(x)
            ce_loss = tf.reduce_sum(-tf.log(p+1e-8), axis=[1])

            rnd_ps = [self.add_noise(x) for _ in range(1)]
            # reg_loss = tf.add_n([tf.reduce_sum(p_-tf.stop_gradient(p), axis=[1]) for p_ in rnd_ps])
            reg_loss = tf.add_n([tf.reduce_sum(-tf.log(1-p_+1e-8), axis=[1]) for p_ in rnd_ps])
            return tf.reduce_mean(reg_loss), tf.reduce_mean(ce_loss)

    @staticmethod
    def preprocess(x):
        im = np.reshape(x, [-1, 28, 28, 1])
        im = np.round(im).astype(np.float32)  # NOTE important !?
        return np.pad(im, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

    def add_noise(self, x):
        x_ = x + tf.random_normal(tf.shape(x))
        p_ = self.cnn(x_)
        return p_
