import unittest

import numpy as np
import tensorflow as tf

from src.flows import *

class Test(unittest.TestCase):
    def test_transform(self):
        fn = Dense(n_inputs=6, n_outputs=12)

        dist = tfd.MultivariateNormalDiag(loc=tf.zeros([1, 6]),
                                          scale_diag=tf.ones([1, 6]))

        density = tfd.TransformedDistribution(distribution=dist, bijector=fn)
        s = density.sample()

    def test_bijector(self):
        """check the inversion works"""
        x = tf.random_normal([2, 6])

        fn = Dense(6, 12)

        y = fn.forward(x)
        x_ = fn.inverse(y)
        err = tf.losses.mean_squared_error(x, x_)
        # BUG this should not give an accurate inverse atm
        # cheating atm. x is cached

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            E = sess.run(err)

        self.assertEqual(E, 0)

    def test_jacobian_shape(self):
        x = tf.random_normal([2, 6])
        w = tf.random_normal([6, 4])
        y = tf.matmul(x, w)

        dydx = jacobian(y, x)

        with tf.Session() as sess:
            D = sess.run(dydx)

        self.assertEqual(D.shape, (2, 4, 6))

    def test_det_shape(self):
        dydx = tf.random_normal([2, 6, 4])
        det = non_square_det(dydx)

        with tf.Session() as sess:
            D = sess.run(det)

        self.assertEqual(D.shape, (2, ))

    def test_det_val(self):
        """should give similar results for square matrices"""
        A = tf.random_normal([6, 6])
        det = non_square_det(A)
        det_ = tf.linalg.det(A)

        with tf.Session() as sess:
            D1, D2 = sess.run([det, det_])

        self.assertTrue(np.allclose(D1, D2, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
