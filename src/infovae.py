import tensorflow as tf
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
    Compute the disctance between x and y using a guassian kernel.
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

class InfoVAE():
    def __init__(self, n_hidden, width, depth):
        """
        Args:

        """
        self.n_hidden = n_hidden
        self.width = width
        self.depth = depth
        self.n_channels = 1
        self.stddev = 0.1

        self.construct()


    def construct(self):
        """
        Constructs:
            encoder (tf.keras.Model): encode the gradient into the hidden space
            decoder (tf.keras.Model): decodes a hidden state into an image
        """
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.width, 4, strides=(2, 2), padding='same', input_shape=(28,28,1)),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2D(self.width, 4, strides=(2, 2), padding='same'),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2D(self.n_hidden*2, 1, strides=(1, 1), padding='same'),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(self.width, 4, strides=(2, 2), padding='same', input_shape=(7,7,self.n_hidden)),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2DTranspose(self.width, 4, strides=(2, 2), padding='same'),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2DTranspose(self.n_channels*2, 1, strides=(1, 1), padding='same'),
        ])

    def __call__(self, x):
        """
        Args:
            x (tf.tensor): the input
                shape is [None, width, height, channels],
                dtype is tf.float32
        """
        h = self.encoder(x)
        self.z = reparameterise(h, self.n_hidden, self.stddev)
        y = self.decoder(self.z)
        return reparameterise(y, self.n_channels, self.stddev)

    def make_losses(self, x, y=None):
        if y is None:
            y = self.__call__(x)

        recon_loss = tf.losses.sigmoid_cross_entropy(
            logits=tf.layers.flatten(y),
            multi_class_labels=tf.layers.flatten(x))
        latent_loss = compute_mmd(tf.layers.flatten(self.z),
                              tf.layers.flatten(tf.random_normal(shape=tf.shape(self.z))))

        return recon_loss, latent_loss

    @staticmethod
    def preprocess(x):
        return x.reshape((-1, 28, 28, 1))

if __name__ == '__main__':
    tf.enable_eager_execution()
    x = tf.random_normal((100, 28, 28, 1))

    nn = InfoVAE(12, 16, 3)
    x_ = nn(x)

    # loss = nn.make_losses(x)

    assert x_.shape == x.shape
