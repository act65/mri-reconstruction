import tensorflow as tf
import src.infovae as infovae

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
    parser.add_argument('--logdir', type=str, default='/tmp/infovae/',
                        help='location to save logs')
    parser.add_argument('--n_hidden', type=int, default=12)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=4)
    return parser.parse_args()

def main(args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)


    nn = infovae.InfoVAE(args.n_hidden, args.width, args.depth)

    x = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)
    x_ = nn(x)

    recon_loss, latent_loss = nn.make_losses(x, x_)
    loss = recon_loss+latent_loss

    train_summaries = [
        tf.summary.scalar('train/loss/recon', recon_loss),
        tf.summary.scalar('train/loss/latent', latent_loss),
        tf.summary.histogram('latents', nn.z),
        tf.summary.image('train/input', x),
        tf.summary.image('train/recon', tf.nn.sigmoid(x_)),
    ]
    test_summaries = [
        tf.summary.scalar('test/loss/recon', recon_loss),
        tf.summary.scalar('test/loss/latent', latent_loss),
        tf.summary.image('test/recon', tf.nn.sigmoid(x_)),
    ]

    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        for i in range(args.epochs*1000):
            batch_x, _ = mnist.train.next_batch(args.batch_size)
            _, train_summ = sess.run([train_step, train_merged],
                                     feed_dict={x: infovae.InfoVAE.preprocess(batch_x)})

            if i % 10 == 0:
                writer.add_summary(train_summ, i)

            if i % 100 == 0:
                L, test_summ = sess.run([loss, test_merged],
                                                    feed_dict={x:
                            infovae.InfoVAE.preprocess(mnist.test.images[:100, ...])})
                print('\rStep: {} Loss: {}'.format(i, L), end='', flush=True)
                writer.add_summary(test_summ, i)
        save_path = saver.save(sess, os.path.join(args.logdir,"infovae.ckpt"))
        print(save_path)

if __name__ == '__main__':
    main(argumentparser())
