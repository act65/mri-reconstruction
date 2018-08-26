import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import src.utils as utils

def gaussian_d(x, y):
    """
    A conceptual lack of understanding here.
    Do I need a dx to calculate this over?
    Doesnt make sense for a single point!?
    """
    d = tf.norm(x - y, axis=1)
    return tf.exp(-0.5*d)/(tf.sqrt(2*tf.constant(np.pi)))

class SparseAE():
    def __init__(self, n_hidden, width, depth, stddev=0.0001):
        """
        Args:

        """
        self.n_hidden = n_hidden
        self.width = width
        self.depth = depth
        self.n_channels = 1
        self.stddev = stddev

        self.e = 1e-1

        self.prior = tfp.distributions.RelaxedBernoulli(1e-1,
                        logits=tf.get_variable(name='prior_params'),
                        shape=[16, 1, 1, n_hidden])

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
        with tf.name_scope('sparseae'):
            self.h = self.encoder(x)
            self.z = tf.nn.relu(self.h + self.e) - tf.nn.relu(-self.h-self.e)
            self.y = self.decoder(self.z)
            return self.y

    def make_losses(self, x, y=None):
        self.x = x
        if y is None:
            print('...')
            y = self.__call__(self.x)

        with tf.name_scope('loss'):
            recon_loss = tf.losses.mean_squared_error(x, y)
            latent_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.z), axis=[1]))
            # or could use a distribution over discrete vals and entropy!?
        return recon_loss, latent_loss

    def estimate_density(self, x):
        x_ = self.__call__(x)
        return gaussian_d(self.z, tf.zeros_like(self.z))


    @staticmethod
    def preprocess(x):
        im = np.reshape(x, [-1, 28, 28, 1])
        im = np.round(im).astype(np.float32)  # NOTE important !?
        return np.pad(im, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

if __name__ == '__main__':
    tf.enable_eager_execution()
    x = tf.random_normal((100, 32, 32, 1))

    nn = SparseAE(12, 16, 4)
    x_ = nn(x)

    loss = nn.make_losses(x)

    assert x_.shape == x.shape
