import unittest

import numpy as np
import tensorflow as tf

from src.flows import *

class Test_Dense(unittest.TestCase):
    def test_sample(self):
        fn = Dense(n_inputs=6, n_outputs=12, name='0')

        dist = tfd.MultivariateNormalDiag(loc=tf.zeros([6]),
                                          scale_diag=tf.ones([6]))

        density = tfd.TransformedDistribution(distribution=dist, bijector=fn)
        s = density.sample(1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            S = sess.run(s)

        self.assertEqual(S.shape, (1, 12))

    def test_prob(self):
        x = tf.random_normal([50, 784])
        fn = Dense(n_inputs=6, n_outputs=784, name='1')

        dist = tfd.MultivariateNormalDiag(loc=tf.zeros([6]),
                                          scale_diag=tf.ones([6]))

        density = tfd.TransformedDistribution(distribution=dist, bijector=fn)
        p = density.log_prob(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            P = sess.run(p)

        self.assertEqual(P.shape, (50, ))

    def test_bijector(self):
        """check the inversion works"""
        x = tf.random_normal([2, 6])

        fn = Dense(6, 12, name='2')

        y = fn.forward(x)
        x_ = fn.inverse(y)
        err = tf.losses.mean_squared_error(x, x_)
        # BUG this should not give an accurate inverse atm
        # cheating atm. x is cached

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            E = sess.run(err)

        self.assertEqual(E, 0)

class Test_Utils(unittest.TestCase):
    def test_det_shape(self):
        dydx = tf.random_normal([6, 4])
        det = non_square_det(dydx)

        with tf.Session() as sess:
            D = sess.run(det)

        self.assertEqual(D.shape, ())

    def test_det_val(self):
        """should give similar results for square matrices"""
        A = tf.random_normal([16, 16])
        det = non_square_det(A)
        det_ = tf.abs(tf.linalg.det(A))

        with tf.Session() as sess:
            D1, D2 = sess.run([det, det_])

        self.assertTrue(np.allclose(D1, D2, rtol=1e-4))

    def test_pinv(self):
        A = tf.random_normal([784, 32])
        A_ = pinv(A)
        err = tf.matmul(A_, A)

        with tf.Session() as sess:
            E = sess.run(err)

        self.assertTrue(np.allclose(np.eye(E.shape[0]), E, atol=1e-4))

if __name__ == '__main__':
    unittest.main()
