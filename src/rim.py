import tensorflow as tf
import utils

"""
Implementation of Recurrent inference machines
https://openreview.net/forum?id=HkSOlP9lg
"""
# TODO what does the hidden space encode?
# TODO vector field of direction of gradient

class RIM(tf.keras.Model):
    def __init__(self, dLdx, units):
        """
        Args:
            dLdx (func): func: tf.tensor -> tf.tensor.
                the gradient of L w.r.t the forward process
            units (int): the number of units in the hidden state
        """
        super(self.__class__, self).__init__(self)
        self.dLdx = dLdx
        self.units = units
        self.construct()

    def construct(self):
        """
        Constructs:
            cnn1 (tf.keras.Model): encode the gradient into the hidden space
            cnn2 (tf.keras.Model): encodes an image into the hidden space
            dcnn (tf.keras.Model): decodes a hidden state into an image
            rnn (tf.keras.Model): a RNN for ...?
        """
        self.cnn1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 4, strides=(2, 2), padding='same'),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2D(4, 4, strides=(2, 2), padding='same'),
        ])
        self.cnn2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, 4, strides=(2, 2), padding='same'),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2D(4, 4, strides=(2, 2), padding='same'),
        ])

        self.dcnn = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, 4, strides=(2, 2), padding='same'),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2DTranspose(32, 4, strides=(2, 2), padding='same'),
            tf.keras.layers.Activation(tf.keras.activations.selu),
            tf.keras.layers.Conv2DTranspose(1, 1, strides=(1, 1), padding='same')
        ])

        self.rnn = tf.keras.layers.RNN([
            tf.keras.layers.GRUCell(self.units),
            tf.keras.layers.GRUCell(self.units)
        ])

    def call(self, y, iters=10):
        """
        Args:
            y (tf.tensor): the k-space samples
                shape is [None, width, height, channels],
                dtype is tf.complex64
            iters (int): the number of recurrent iterations, aka the steps of
                gradient descent on x
        """
        # TODO how is this trained with big images!? not possible!?
        # the recurrent dependencies will use up lots of memory!?!

        x_t = tf.ifft2d(y)  # init x_0

        self.candidate_xs = [x_t]

        for i in range(iters):
            g = self.dLdx(y, x_t)

            g_e = self.embed(g, self.cnn1)
            x_t_e = self.embed(x_t, self.cnn2)

            # TODO does g_e actually provide useful information!?
            h = self.encoder(g_e, x_t_e)
            x_t += self.decoder(g_e, x_t_e, h)

            self.candidate_xs.append(x_t)

        return x_t

    def embed(self, x, cnn):
        x = utils.complex2float(x)
        x = cnn(x)
        self.shape = x.shape
        return tf.reshape(x, (x.shape[0], -1))

    def encoder(self, g, x_t):
        x = tf.concat([g, x_t], axis=-1)
        x = tf.reshape(x, [x_t.shape[0], 1, -1])

        y = self.rnn(x)
        return y

    def decoder(self, g, x_t, h, agg='add'):
        # TODO will need to experiment with these?
        if agg == 'add':
            x = g + x_t + h
            x = tf.reshape(x, self.shape)
        elif agg == 'concat':
            g = tf.reshape(g, self.shape)
            x_t = tf.reshape(x_t, self.shape)
            h = tf.reshape(h, [h.shape[0], 1, 1, h.shape[1]])
            x = tf.concat([g, x_t], axis=-1) + h

        y = self.dcnn(x)
        y = tf.cast(y, tf.complex64)
        return y


if __name__ == '__main__':
    from mri import dLdx
    from utils import complex_random

    y = complex_random(tf.random_normal)(shape=(10, 28, 28, 1))

    rim = RIM(dLdx, 196)
    x_t = rim(y)

    assert x_t.shape == y.shape
