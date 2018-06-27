import tensorflow as tf
import src.classifier as classifier

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
    mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=False)


    nn = classifier.Classifier(args.n_hidden, args.width, args.depth)

    x = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)
    labels = tf.placeholder(shape=[None], dtype=tf.int32)
    logits = nn(x)

    loss = nn.make_losses(x, labels)

    l = tf.argmax(logits,1)
    acc, acc_op = tf.metrics.accuracy(labels=labels,
                                  predictions=l)
    conf_mat = tf.cast(tf.confusion_matrix(labels, l), tf.float32)
    fp, fp_op = tf.metrics.false_negatives(labels, l)
    fn, fn_op = tf.metrics.false_positives(labels, l)
    print(fp, fn)

    # want per label false positives/negatives.

    train_summaries = [
        tf.summary.scalar('train/loss', loss),
        tf.summary.histogram('latents', logits),
        tf.summary.scalar('train/acc', acc),
        tf.summary.image('confusion', tf.expand_dims(tf.expand_dims(conf_mat, 0), -1))
    ]
    test_summaries = [
        tf.summary.scalar('test/loss', loss),
        tf.summary.scalar('test/acc', acc),
        tf.summary.scalar('test/fp', fp),
        tf.summary.scalar('test/fn', fn)
    ]

    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss)
    train_step = tf.group(*[train_step, acc_op, fp_op, fn_op])
    # saver = tf.train.Saver()

    checkpoint = tf.contrib.eager.Checkpoint(**{var.name: var for var in tf.global_variables()})

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        for i in range(args.epochs*1000):
            batch_x, batch_l = mnist.train.next_batch(args.batch_size)
            _, train_summ = sess.run([train_step, train_merged],
                                     feed_dict={x: classifier.Classifier.preprocess(batch_x),
                                                labels: batch_l})

            if i % 10 == 0:
                writer.add_summary(train_summ, i)

            if i % 100 == 0:
                L, test_summ = sess.run([loss, test_merged],
                                                    feed_dict={x:
                            classifier.Classifier.preprocess(mnist.test.images[:100, ...]),
                            labels: mnist.test.labels[:100, ...]})
                print('\rStep: {} Loss: {}'.format(i, L), end='', flush=True)
                writer.add_summary(test_summ, i)
                sess.run(tf.local_variables_initializer())

        save_path = checkpoint.save(os.path.join(args.logdir,"infovae.ckpt"))
        # save_path = saver.save(sess, os.path.join(args.logdir,"infovae.ckpt"))
        print(save_path)

if __name__ == '__main__':
    main(argumentparser())
