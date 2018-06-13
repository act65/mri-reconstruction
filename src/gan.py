import tensorflow as tf
import numpy as np
import src.utils as utils

class GAN():
    def __init__(self, n_hidden, width, depth):
        """
        Args:

        """
        self.n_hidden = n_hidden
        self.width = width
        self.depth = depth

        self.n_channels = 1

        self.construct()

    def construct(self):
        """
        Constructs:
            discriminator (tf.keras.Model): produces a binary decision given an image
            generator (tf.keras.Model): generates an image from some input
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
        layers.append(tf.keras.layers.Conv2D(1,
                                1,
                                strides=(1, 1),
                                padding='same'))
        self.discriminator = tf.keras.Sequential(layers)

        # generator
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
        self.generator = tf.keras.Sequential(layers)

    def __call__(self, x, z=None):
        """
        Args:
            x (tf.tensor): the input
                shape is [None, width, height, channels],
                dtype is tf.float32
        """
        if z is None:
            z = tf.random_normal(shape=(tf.shape(x)[0], 1, 1, self.n_hidden))

        x_fake = self.generator(z)
        p_real = self.discriminator(x)
        p_fake = self.discriminator(x_fake)
        return x_fake, tf.layers.flatten(p_real), tf.layers.flatten(p_fake)

    def make_losses(self, x, x_fake=None, p_real=None, p_fake=None):
        if x_fake is None and p_real is None and p_fake is None:
            x_fake, p_real, p_fake = self.__call__(x)

        # discriminator is trying to correctly guess fake/real labelss
        discrim_loss = tf.losses.sigmoid_cross_entropy(
            logits=p_real,
            multi_class_labels=tf.ones_like(p_real))
        discrim_loss += tf.losses.sigmoid_cross_entropy(
            logits=p_fake,
            multi_class_labels=tf.zeros_like(p_fake))

        # generator is trying to fool the discriminator
        gen_loss = tf.losses.sigmoid_cross_entropy(
            logits=p_fake,
            multi_class_labels=tf.ones_like(p_fake))

        return gen_loss, discrim_loss

    @staticmethod
    def preprocess(x):
        im = x.reshape((-1, 28, 28, 1))
        return np.pad(im, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

if __name__ == '__main__':
    # tf.enable_eager_execution()
    x = tf.random_normal((100, 32, 32, 1))

    nn = GAN(12, 16, 4)
    x_fake, p_real, p_fake = nn(x)
    print(x_fake, p_real, p_fake)

    gen_loss, discrim_loss = nn.make_losses(x)
    print(gen_loss, discrim_loss)
