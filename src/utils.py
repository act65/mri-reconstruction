import tensorflow as tf

def complex2float(x):
    return tf.concat([tf.real(x), tf.imag(x)], axis=-1)

def complex_random(init):
    def func(shape, dtype=None, partition_info=None):
        real = init(dtype=tf.float32, shape=shape)
        imag = init(dtype=tf.float32, shape=shape)
        return tf.complex(real, imag)
    return func

class PyramidLoss(tf.keras.Model):
    def __init__(self, width, depth):
        super(self.__class__, self).__init__(self)
        self.conv_layers = [
            tf.keras.layers.Conv2D(width,
                                   4,
                                   strides=(2, 2),
                                   padding='same',
                                   activation=tf.keras.activations.selu)
                            for i in range(depth)]

        self.loss_weights = [1] + [1 for i in range(depth)]

    def call(self, inputs):
        y, t = inputs
        y_a = self.apply_convs(y)
        t_a = self.apply_convs(t)

        return tf.add_n([w*tf.losses.mean_squared_error(a, b)
                         for w, a, b in zip(self.loss_weights, y_a, t_a)])


    def apply_convs(self, x):
        activations = [x]
        for conv in self.conv_layers:
            x = conv(x)
            activations.append(x)
        return activations

def input_fn(batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    """
    ds = input_fn(50)
    iterator = ds.make_one_shot_iterator()
    img, t = iterator.get_next()
    init_op = iterator.make_initializer(ds)
    """

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=False)

    dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels))
    dataset = dataset.map(lambda x, y: (tf.reshape(x, [28, 28, 1]), tf.cast(y, tf.int32)))
    dataset = dataset.map(lambda x, y: (tf.pad(x, [[2,2], [2,2], [0,0]], "CONSTANT"), tf.reshape(y, [-1])))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

if __name__ == '__main__':
    t = tf.random_normal(shape=(10, 32, 32, 1))
    y = tf.random_normal(shape=(10, 32, 32, 1))

    loss = PyramidLoss(32, 3)
    loss((y,t))
