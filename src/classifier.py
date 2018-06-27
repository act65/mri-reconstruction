import tensorflow as tf
import numpy as np

class Classifier():
    def __init__(self, n_hidden, width, depth):
        self.n_hidden = n_hidden
        self.width = width
        self.depth = depth

        self.construct()


    def construct(self):
        """
        Constructs:
            cnn (tf.keras.Model): encode the gradient into the hidden space
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
        layers.append(tf.keras.layers.GlobalAveragePooling2D())
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.Dense(10))
        self.encoder = tf.keras.Sequential(layers)

    def __call__(self, x):
        """
        Args:
            x (tf.tensor): the input
                shape is [None, width, height, channels],
                dtype is tf.float32
        """
        with tf.name_scope('classifier'):
            return self.encoder(x)

    def make_losses(self, x, y):
        with tf.name_scope('loss'):
            return tf.losses.sparse_softmax_cross_entropy(
                logits=self.__call__(x),
                labels=y)

    @staticmethod
    def preprocess(x):
        im = np.reshape(x, [-1, 28, 28, 1])
        im = np.round(im).astype(np.float32)  # NOTE important !?
        return np.pad(im, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

if __name__ == '__main__':
    # tf.enable_eager_execution()
    x = tf.random_normal((100, 32, 32, 1))
    l = tf.random_uniform(minval=0, maxval=10, shape=[100], dtype=tf.int32)

    nn = Classifier(12, 16, 3)
    logits = nn(x)
    assert logits.shape == [100, 10]

    loss = nn.make_losses(x, l)
    print(loss)
