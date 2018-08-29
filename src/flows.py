import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

def jacobian(y, x):
    """
    Code adapted from https://github.com/tensorflow/tensorflow/issues/675#issuecomment-319891923

    Args:
        y: A 2d tf.tensor [batch, n_outputs]
        x: A 2d tf.tensor [batch, n_inputs]

    Returns:
        [batch, n_outputs, n_inputs]
    """
    y_flat = tf.reshape(y, [-1])
    n = y_flat.get_shape()[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)

    jacobian = jacobian.stack()

    dydx = tf.reshape(jacobian, y.shape.concatenate(x.shape))
    return tf.reduce_sum(dydx, axis=[2])

def non_square_det(x):
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
    squared_mat = tf.matmul(x, x, transpose_b=True)
    return tf.sqrt(tf.linalg.det(squared_mat))

class Dense(tfb.Bijector):
    """
    Want a hierarchical flow.
    Map some low dim distribution to a manifold in a higher dimensional space.
    """
    def __init__(self, n_inputs, n_outputs, validate_args=False, name='dense'):
        """
        Args:
            n_inputs (int): the number of features (last dim)
            n_outputs (int): the target num of feautres
        """
        super(self.__class__, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=1,
            name=name)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(name='weights',
                                           shape=[n_inputs, n_outputs],
                                           dtype=tf.float32)
            self.bias = tf.get_variable(name='bias',
                                        shape=[n_outputs],
                                        dtype=tf.float32)

    def _forward_event_shape_tensor(self, shape):
        return tf.shape([shape[0], self.n_inputs])

    def _invserse_event_shape_tensor(self, shape):
        return tf.shape([shape[0], self.n_outputs])

    def _forward(self, x):
        return tf.matmul(x, self.weights) + self.bias

    def _inverse(self, y):
        # BUG this is only true if the weights are unitary
        # need to fix, maybe be use approx/learned inversion
        return tf.matmul(y - self.bias, self.weights, transpose_b=True)

    def _forward_log_det_jacobian(self, x):
        y = self.forward(x)
        J = jacobian(y, x)
        return tf.log(non_square_det(J))

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self.inverse(y))
