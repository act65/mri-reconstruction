import tensorflow as tf
from functools import partial
import src.utils as utils

"""
Here we encode our knowledge about the forward process, in this case MRI.
"""
# TODO actual sampling of k-space occurs row-column wise and/or in spirals

class MRI():
    def __init__(self, stddev=1.0, n=4):
        self.stddev = stddev
        self.n = n

    def __call__(self, x):
        """
        The forward process of a MRI scan is supposedly well understood.

        Args:
            x (tf.tensor): The input image
                shape is [None, width, height, channels],
                dtype is tf.float32

        Returns:
            y (tf.tensor): The outputs in k-space
                shape is [None, width, height, channels],
                dtype is tf.complex 64
        """
        y = tf.fft2d(x)

        # generate a random mask. aka the samples from y that we choose
        mask = tf.random_uniform(tf.shape(y), minval=0, maxval=self.n, dtype=tf.int32)
        mask = 1-tf.cast(tf.greater(mask, tf.ones_like(mask)), tf.float32)
        self.mask = tf.complex(mask, mask)
        y *= self.mask

        # also add some noise
        y += utils.complex_random(tf.random_normal)(tf.shape(y))*self.stddev

        return y

    def dLdx(self, y, x, sigma=1.0):
        # TODO derivation and validation
        """
        Args:
            x (tf.tensor): The input image
                shape is [None, width, height, channels],
                dtype is tf.float32
            y (tf.tensor): The outputs in k-space
                shape is [None, width, height, channels],
                dtype is tf.complex64

        Returns:
            (tf.tensor): the grad of the loss w.r.t x
                shape is [None, width, height, channels],
                dtype is tf.complex64
        """
        y_ = self.mask*tf.fft(x)
        return tf.ifft2d(y_-y)/(sigma**2)
