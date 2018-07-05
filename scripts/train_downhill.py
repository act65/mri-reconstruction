import tensorflow as tf
from src.downhill_density import Density

import os
import argparse

def argumentparser():
    parser = argparse.ArgumentParser(description='Train an InfoVAE')
    parser.add_argument('--num_units', type=int, default=64,
                        help='Dimension of each world')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size...')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--logdir', type=str, default='/tmp/downhill/',
                        help='location to save logs')
    parser.add_argument('--n_hidden', type=int, default=12)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--beta', type=float, default=0.1)
    return parser.parse_args()


def main(args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)

    # ds = input_fn(mnist.train.images, mnist.train.labels, 50)
    # iterator = ds.make_one_shot_iterator()
    # x, t = iterator.get_next()
    # init_op = iterator.make_initializer(ds)

    x = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)

    nn = Density(args.n_hidden, args.width, args.depth)
    p = nn(x)
    reg_loss, ce_loss = nn.make_losses(x)
    loss = args.beta*reg_loss + ce_loss

    p_rnd = nn(tf.random_normal(tf.shape(x)))

    labels = tf.concat([p, p_rnd], axis=0)
    truth = tf.concat([tf.ones_like(p), tf.zeros_like(p_rnd)], axis=0)
    labels = tf.argmax(labels,1)
    acc, acc_op = tf.metrics.accuracy(labels=truth,
                                      predictions=labels)

    train_summaries = [
        tf.summary.scalar('train/ce_loss', ce_loss),
        tf.summary.scalar('train/reg_loss', reg_loss),
        tf.summary.scalar('train/acc', acc),
        tf.summary.histogram('predictions', labels)
    ]
    test_summaries = [
        tf.summary.scalar('test/loss', loss),
        tf.summary.scalar('test/acc', acc),
    ]

    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)

    opt = tf.train.AdamOptimizer()
    # opt = tf.train.MomentumOptimizer(0.01, 0.9)
    train_step = opt.minimize(loss)
    train_step = tf.group(*[train_step, acc_op])
    # saver = tf.train.Saver()

    checkpoint = tf.contrib.eager.Checkpoint(**{var.name: var for var in tf.global_variables()})

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        for i in range(args.epochs*1000):
            batch_x, batch_l = mnist.train.next_batch(args.batch_size)
            _, train_summ = sess.run([train_step, train_merged],
                                     feed_dict={x: Density.preprocess(batch_x)})

            if i % 10 == 0:
                writer.add_summary(train_summ, i)

            if i % 100 == 0:
                L, test_summ = sess.run([loss, test_merged],
                                                    feed_dict={x:
                            Density.preprocess(mnist.test.images[:100, ...])})
                print('\rStep: {} Loss: {}'.format(i, L), end='', flush=True)
                writer.add_summary(test_summ, i)
                sess.run(tf.local_variables_initializer())

        save_path = checkpoint.save(os.path.join(args.logdir,"infovae.ckpt"))
        # save_path = saver.save(sess, os.path.join(args.logdir,"infovae.ckpt"))
        print(save_path)

if __name__ == '__main__':
    main(argumentparser())
