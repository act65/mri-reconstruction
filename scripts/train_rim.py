import tensorflow as tf
import src.rim as rim

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
    parser.add_argument('--logdir', type=str, default='/tmp/gan/',
                        help='location to save logs')
    parser.add_argument('--n_hidden', type=int, default=12)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=4)
    return parser.parse_args()

def summarise_candidates(model):
    for i, x_t in enumerate(model.candidate_xs):
        tf.summary.image('x_t/{}'.format(i), tf.real(x_t), max_outputs=1)

def candidate_loss(model, x):
    # new idea. can introduce a loss at every step, rather than just at
    # the end. need to test versus loss at end.
    return tf.add_n([tf.losses.mean_squared_error(x, x_t)
              for x_t in model.candidate_xs])

def main(args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)

    f = mri.MRI()
    y = f(x)

    rim = rim.RIM(f.dLdx, 196)
    x_t = rim(y)

    loss = candidate_loss(rim, x)

    x, x_t = (tf.real(x), tf.real(x_t))

    summarise_candidates(rim)
    tf.summary.scalar('loss', loss),
    merged = tf.summary.merge_all()

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        for i in range(args.epochs*1000):
            batch_x, _ = mnist.train.next_batch(args.batch_size)
            _, train_summ = sess.run([train_step, merged],
                                     feed_dict={x: rim.RIM.preprocess(batch_x)})

            if i % 10 == 0:
                writer.add_summary(train_summ, i)

            if i % 100 == 0:
                L, test_summ = sess.run([gen_loss, test_merged],
                                                    feed_dict={x:
                            rim.RIM.preprocess(mnist.test.images[:100, ...])})
                print('\rStep: {} Loss: {}'.format(i, L), end='', flush=True)
                writer.add_summary(test_summ, i)

        save_path = saver.save(sess, os.path.join(args.logdir,"rim.ckpt"))
        print(save_path)

if __name__ == '__main__':
    main(argumentparser())
