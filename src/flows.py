import os
import numpy as np
import urllib

from absl import flags

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "epochs", default=100, help="Number of training steps to run.")
flags.DEFINE_string(
    "activation",
    default="selu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_string(
    "data_dir",
    default="/tmp/mnist",
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default="/tmp/critic/",
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=500, help="Frequency at which to save visualizations.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS

def non_square_det(x, reltol=1e-6):
    """
    Idea taken from https://www.quora.com/How-do-we-calculate-the-determinant-of-a-non-square-matrix

    # for n != m
    A = tf.random_normal([n, m])
    det(A) := sqrt(det(A.A^T))

    Args:
        x (tf.tensor): shape in [..., a, b]

    Returns:
        [..., ]
    """
    # squared_mat = tf.matmul(x, x, transpose_b=True)
    # return tf.sqrt(tf.linalg.det(squared_mat))

    s = tf.svd(x, compute_uv=False)

    # atol = tf.reduce_max(s) * reltol
    # s = tf.diag(tf.where(tf.greater(atol, tf.abs(s)), tf.ones_like(s), s))

    return tf.reduce_prod(s)

def pinv(A, reltol=1e-6):
    """
    Args:
        A (tf.tensor): the matrix to be inverted shape=[n, m]

    Returns:
        inverse (tf.tensor): the invserse of A, s.t. A_T.A = I. shape=[m,n]
    """
    s, u, v = tf.svd(A)

    atol = tf.reduce_max(s) * reltol
    s_inv = tf.diag(tf.where(tf.greater(tf.abs(s), atol), 1.0/s, tf.zeros_like(s)))
    # s_inv = tf.diag(1./s)

    return tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

class Dense(tfb.Bijector):
    """
    Want a hierarchical flow.
    Map some low dim distribution to a manifold in a higher dimensional space.
    """
    def __init__(self, n_inputs, n_outputs, validate_args=False, name=''):
        """
        Args:
            n_inputs (int): the number of features (last dim)
            n_outputs (int): the target num of feautres
        """
        super(self.__class__, self).__init__(
            validate_args=validate_args,
            is_constant_jacobian=True,
            forward_min_event_ndims=1,
            name=name)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        with tf.variable_scope('dense'+name):
            self.weights = tf.get_variable(name='weights',
                                           shape=[n_inputs, n_outputs],
                                           dtype=tf.float32,
                                           # initializer=tf.initializers.orthogonal()
                                           )
            self.bias = tf.get_variable(name='bias',
                                        shape=[n_outputs],
                                        dtype=tf.float32,
                                        initializer=tf.initializers.zeros()
                                        )

    @property
    def _is_injective(self):
        return True

    def _forward_event_shape_tensor(self, shape):
        return tf.shape([shape[0], self.n_inputs])

    def _invserse_event_shape_tensor(self, shape):
        return tf.shape([shape[0], self.n_outputs])

    def _forward(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def _inverse(self, y):
        weights_inv = pinv(self.weights)
        return tf.matmul(y - self.bias, weights_inv)

    def _forward_log_det_jacobian(self, x):
        return tf.log(non_square_det(self.weights))

    def _inverse_log_det_jacobian(self, y):
        return tf.log(non_square_det(pinv(self.weights)))

def model_fn(features, labels, mode, params, config):
    """
    Builds the model function for use in an estimator.
    Arguments:
        features: The input features for the estimator.
        labels: The labels, unused here.
        mode: Signifies whether it is train or test or predict.
        params: Some hyperparameters as a dictionary.
        config: The RunConfig, unused here.
    Returns:
        EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    x = features['x']

    global_step = tf.train.get_or_create_global_step()
    with tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=global_step):

        # construct a multilayer parameterised bijector
        n_hidden = 8
        width = 32
        # fn = Dense(n_hidden, 784)

        fn = tfb.Chain([
            Dense(width, 784, name='3'),
            # tfb.Softplus(),
            # Dense(width, width, name='2'),
            Dense(width, width, name='1'),
            Dense(n_hidden, width, name='0')
        ])

        # use the bijector to map a simple distribution into our a density model
        dist = tfd.MultivariateNormalDiag(loc=tf.zeros([n_hidden]),
                                          scale_diag=tf.ones([n_hidden]))
        density = tfd.TransformedDistribution(distribution=dist, bijector=fn)

        # maximise the likelihood of the data
        p = density.prob(x)
        loss = tf.reduce_mean(1-p) # + 0.1*density.entropy()
        # loss = -density.entropy()

        # generate some samples to visualise
        # HACK to get samples to work I had to comment out line 411 of transformed_distribution.py
        samples = density.sample(3)
        tf.summary.image('samples', tf.reshape(samples, [3, 28, 28, 1]))

        # mu = density.mean()
        # tf.summary.image('mean', tf.reshape(mu, [1, 28, 28, 1]))

        opt = tf.train.AdamOptimizer()
        gnvs = opt.compute_gradients(loss)
        gnvs = [(tf.clip_by_norm(g, 1.0) if g is not None else tf.zeros_like(v), v) for g, v in gnvs]
        train_step = opt.apply_gradients(gnvs, global_step=global_step)

        """
        Problems:
        - low probabilty assigned to images at init (how can we init/regularise for uniform dist?)
        - adding non linearities
        """

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_step,
      eval_metric_ops={"eval_loss": tf.metrics.mean(loss)}
    )

def main(_):
    params = FLAGS.flag_values_dict()
    params["activation"] = getattr(tf.nn, params["activation"])

    if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
        tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": train_data},
          y=train_labels,
          batch_size=FLAGS.batch_size,
          num_epochs=1,
          shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=1,
        shuffle=False)


    estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
    )

    for _ in range(FLAGS.epochs):
        estimator.train(train_input_fn, steps=FLAGS.viz_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)

if __name__ == "__main__":
    tf.app.run()
