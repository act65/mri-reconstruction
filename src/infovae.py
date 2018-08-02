import tensorflow as tf
import numpy as np
import src.utils as utils

"""
Implementation of InfoVAE
https://arxiv.org/abs/1706.02262
"""

def reparameterise(x, n, stddev):
    """
    Model each output as bing guassian distributed.
    Use the reparameterisation trick so we can sample while remaining
    differentiable.
    """
    with tf.name_scope('reparameterise'):
        z_mean = x[:,:,:,:n]
        z_stddev = x[:,:,:,n:]
        e = tf.random_normal(tf.shape(z_mean), stddev=stddev)

        # TODO log_var or stddev?
        return z_mean + z_stddev*e

def compute_kernel(x, y):
    """
    Compute the distance between x and y using a guassian kernel.
    """
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = tf.tile(tf.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    """
    Calculate the maximum mean disrepancy..
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def gaussian_d(x, y):
    """
    A conceptual lack of understanding here.
    Do I need a dx to calculate this over?
    Doesnt make sense for a single point!?
    """
    d = tf.norm(x - y, axis=1)
    return tf.exp(-0.5*d)/(tf.sqrt(2*tf.constant(np.pi)))

def pz(z):
    """
    Estimate p(z) using our prior on z.
    """
    z = tf.layers.flatten(z)
    return gaussian_d(z , tf.zeros_like(z))

def px_z(x_, y):
    # the added noise in the hidden layer.
    return gaussian_d(tf.layers.flatten(y[:,:,:,:1]),
                      tf.layers.flatten(x_))

def pz_x(h, z):
    # the added noise in the final layer.
    shape = h.get_shape().as_list()
    return gaussian_d(tf.layers.flatten(h[:,:,:,:shape[-1]//2]),
                      tf.layers.flatten(z))

def p_bayes(x_, y, h, z):
    """
    If p(z | x) is far away from p(z) then p(x) is low
    p(x) = p(x | z) p(z) / p(z | x)
    """
    return px_z(x_, y) * pz(z) / pz_x(h, z)

# def KL_divergence(p, q):
#     return tf.reduce_sum(p * tf.log(p/q), axis=-1)
#
# def bayesian_surprise(z):
#     """
#
#     """
#     return kl(z, prior)

class InfoVAE():
    def __init__(self, n_hidden, width, depth, stddev=0.01):
        """
        Args:

        """
        self.n_hidden = n_hidden
        self.width = width
        self.depth = depth
        self.n_channels = 1
        self.stddev = stddev

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
        layers.append(tf.keras.layers.Conv2D(self.n_hidden*2,
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
        layers.append(tf.keras.layers.Conv2DTranspose(self.n_channels*2, 1, strides=(1, 1), padding='same'))
        self.decoder = tf.keras.Sequential(layers)

    def __call__(self, x):
        """
        Args:
            x (tf.tensor): the input
                shape is [None, width, height, channels],
                dtype is tf.float32
        """
        with tf.name_scope('infovae'):
            self.h = self.encoder(x)
            self.z = reparameterise(self.h, self.n_hidden, self.stddev)
            self.y = self.decoder(self.z)
            self.x_ = reparameterise(self.y, self.n_channels, self.stddev)
            return self.x_

    def make_losses(self, x, y=None):
        self.x = x
        if y is None:
            print('...')
            y = self.__call__(self.x)

        with tf.name_scope('loss'):
            recon_loss = tf.losses.sigmoid_cross_entropy(
                logits=tf.layers.flatten(y),
                multi_class_labels=tf.layers.flatten(self.x))
            latent_loss = compute_mmd(tf.layers.flatten(self.z),
                                  tf.layers.flatten(tf.random_normal(shape=tf.shape(self.z))))

        return recon_loss, latent_loss

    def make_contractive_loss(self):
        # assumes make_losses has already been called
        print(self.h, self.x)
        dhdx = tf.gradients(self.h, self.x)[0]
        print(dhdx)
        if dhdx is None:
            raise ValueError()
        return tf.reduce_mean(tf.reduce_sum(tf.square(dhdx), axis=[1,2,3]))

    def estimate_density(self, x):
        x_ = self.__call__(x)
        return p_bayes(x_, self.y, self.h, self.z)

    @staticmethod
    def preprocess(x):
        im = np.reshape(x, [-1, 28, 28, 1])
        im = np.round(im).astype(np.float32)  # NOTE important !?
        return np.pad(im, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

if __name__ == '__main__':
    tf.enable_eager_execution()
    x = tf.random_normal((100, 28, 28, 1))

    nn = InfoVAE(12, 16, 3)
    x_ = nn(x)

    # loss = nn.make_losses(x)

    assert x_.shape == x.shape
