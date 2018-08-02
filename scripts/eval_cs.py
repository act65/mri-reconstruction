import tensorflow as tf
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

def reconstruct(y, regulariser):
    pass

def l1(x):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(x), axis=1))

def tv(x):
    return tf.reduce_mean(tf.image.total_variation(x))

def main(args):
    x, labels =

    if args.regulariser == 'l1':
        reg = l1
    else:
        reg = tv

    y = mri(x)
    recon = reconstruct(y, reg)
    loss = tf.losses.mean_squared_error(recon, x)

    summaries = [
        tf.summary.scalar('loss', loss),
    ]

    merged = tf.summary.merge(summaries)
    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss)

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
