import tensorflow as tf

"""
Here we encode our knowledge about the forward process, in this case MRI.
"""
# TODO actual sampling of k-space occurs row-column wise and/or in spirals

def mri(x):
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
    init_fn = partial(tf.random_uniform, minval=0, maxval=1)
    sub_sampler = complex_random(init_fn)(tf.shape(y))
    y *= sub_sampler

    # also add some noise
    y += complex_random(tf.random_normal)(tf.shape(y))

    return y

def dLdx(y, x, sigma=1.0):
    # TODO derivation and validation
    """
    Args:
        x (tf.tensor): The input image
            shape is [None, width, height, channels],
            dtype is tf.float32
        y (tf.tensor): The outputs in k-space
            shape is [None, width, height, channels],
            dtype is tf.complex 64

    Returns:
        (tf.tensor): ???
    """
    # TODO add mask, but needs to be the same one used to generate y
    y_ = tf.fft(x)
    return tf.ifft2d(y_-y)/(sigma**2)
