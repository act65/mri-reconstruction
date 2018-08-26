import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import src.utils as utils

class BernoulliAE():
    def __init__(self, n_hidden, width, depth, stddev=0.0001, batch_size=16):
        """
        Args:

        """
        self.n_hidden = n_hidden
        self.width = width
        self.depth = depth
        self.n_channels = 1
        self.stddev = stddev

        self.temp = 10.0

        self.prior_variables = [tf.get_variable(name='prior_params',
                                          shape=[1, 1, 1, n_hidden],
                                          dtype=tf.float32)]
        self.prior = tfp.distributions.RelaxedBernoulli(self.temp, logits=self.prior_variables[0])

        self.construct()


    def construct(self):
        """
        Constructs:
            encoder (tf.keras.Model): encode the gradient into the hidden space
            decoder (tf.keras.Model): decodes a hidden state into an image
        """
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
        layers.append(tf.keras.layers.Conv2D(self.n_hidden,
                                1,
                                strides=(1, 1),
                                padding='same'))
        self.encoder = tf.keras.Sequential(layers)

        # decoder
        layers = []
        layers.append(tf.keras.layers.Conv2DTranspose(self.width, 4, strides=(2, 2),
                                            padding='same',
                                            # input_shape=(1,1,self.n_hidden)
                                            ))
        layers.append(tf.keras.layers.Activation(tf.keras.activations.selu))
        for _ in range(self.depth):
            layers.append(tf.keras.layers.Conv2DTranspose(self.width, 4, strides=(2, 2), padding='same'))
            layers.append(tf.keras.layers.Activation(tf.keras.activations.selu))
        layers.append(tf.keras.layers.Conv2DTranspose(self.n_channels, 1, strides=(1, 1), padding='same'))
        self.decoder = tf.keras.Sequential(layers)

    def __call__(self, x):
        """
        Args:
            x (tf.tensor): the input
                shape is [None, width, height, channels],
                dtype is tf.float32
        """
        with tf.name_scope('bernoulliae'):
            self.h = self.encoder(x)

            bernoulli = tfp.distributions.RelaxedBernoulli(self.temp, logits=self.h)
            self.z = bernoulli.sample()

            self.y = self.decoder(self.z)
            return self.y

    def make_losses(self, x, y=None):
        self.x = x
        if y is None:
            y = self.__call__(self.x)

        with tf.name_scope('loss'):
            recon_loss = tf.losses.mean_squared_error(x, y)
            prior_loss = tf.reduce_mean(-self.estimate_density(self.z))  # want to maximise p(z | x)
            # or could use a distribution over discrete vals and entropy!?
        return recon_loss, prior_loss

    def estimate_density(self, z):
        p = tf.matmul(tf.layers.flatten(self.prior.probs), tf.layers.flatten(z), transpose_b=True)
        # p = self.prior.prob(self.z)
        return tf.reduce_mean(p)

    @staticmethod
    def preprocess(x):
        im = np.reshape(x, [-1, 28, 28, 1])
        im = np.round(im).astype(np.float32)  # NOTE important !?
        return np.pad(im, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

if __name__ == '__main__':
    tf.enable_eager_execution()
    x = tf.random_normal((16, 32, 32, 1))

    nn = BernoulliAE(12, 16, 4)
    x_ = nn(x)

    loss = nn.make_losses(x)

    assert x_.shape == x.shape
