import tensorflow as tf
import numpy as np
from functools import partial
import src.utils as utils

"""
Here we encode our knowledge about the forward process, in this case MRI.
"""
# TODO actual sampling of k-space occurs row-column wise and/or in spirals

class MRI():
    def __init__(self, stddev=0.1, n=4, N=100):
        self.stddev = stddev
        self.n = n
        self.idx = tf.constant(np.random.choice(32*32*1, N, replace=False))

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
        if x.dtype != tf.complex64:
            x = tf.complex(x, tf.zeros_like(x))

        y = tf.fft2d(x)
        y = tf.concat([tf.real(y) + tf.imag(y)], axis=1)  # cheating?
        # y = tf.sqrt(tf.imag(y)**2 + tf.real(y)**2)

        # TODO not sure this works as intended
        # generate a random mask. aka the samples from y that we choose
        # mask = tf.random_uniform(tf.shape(y), minval=0, maxval=self.n, dtype=tf.int32)
        # mask = 1-tf.cast(tf.greater(mask, tf.ones_like(mask)), tf.float32)
        # self.mask = tf.complex(mask, mask)
        # y *= self.mask

        y = tf.layers.flatten(y)
        y = tf.gather(y, self.idx, axis=1)

        # also add some noise
        y += tf.random_normal(tf.shape(y))*self.stddev

        return y

    def dLdx(self, y, x, sigma=1.0):
        # TODO derive and validate
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
        # gets the mask used in the forward process.
        # NOTE be careful here. will only return the correct mask if called
        # after the corresponding forward process
        y_ = self.mask*tf.fft(x)
        return tf.ifft2d(y_-y)/(sigma**2)
