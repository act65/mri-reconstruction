import tensorflow as tf
import src.gan as gan

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

def main(args):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)


    nn = gan.GAN(args.n_hidden, args.width, args.depth)

    x = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)
    fake, p_real, p_fake = nn(x)

    gen_loss, discrim_loss = nn.make_losses(x, fake, p_real, p_fake)

    train_summaries = [
        tf.summary.scalar('train/loss/gen', gen_loss),
        tf.summary.scalar('train/loss/dicrim', discrim_loss),
        tf.summary.image('train/input', x),
        tf.summary.image('train/gen', fake),
    ]
    test_summaries = [
        tf.summary.scalar('test/loss/gen', gen_loss),
        tf.summary.scalar('test/loss/dicrim', discrim_loss),
        tf.summary.image('test/gen', fake),
    ]

    train_merged = tf.summary.merge(train_summaries)
    test_merged = tf.summary.merge(test_summaries)

    with tf.variable_scope('opt') as scope:
        dis_opt = tf.train.AdamOptimizer(0.0001)#, beta1=0.9, beta2=0.99)
        gen_opt = tf.train.AdamOptimizer(0.0001)#, beta1=0.7, beta2=0.9)
        discrim_gnvs = dis_opt.compute_gradients(discrim_loss, var_list=nn.discriminator.variables)
        gen_gnvs = gen_opt.compute_gradients(gen_loss, var_list=nn.generator.variables)
        discrim_gnvs = [(tf.clip_by_norm(g, 10), v) for g, v in discrim_gnvs]
        gen_gnvs = [(tf.clip_by_norm(g, 10), v) for g, v in gen_gnvs]

        gnvs = discrim_gnvs + gen_gnvs

        dis_step = dis_opt.apply_gradients(discrim_gnvs)
        gen_step = gen_opt.apply_gradients(gen_gnvs)
        train_step = tf.group(*[gen_step, dis_step])

    # print(dir(tf.GraphKeys))
    # print(dir(opt))
    # print(tf.GraphKeys._VARIABLE_COLLECTIONS)
    # print(opt.variables())
    # print(help(opt.get_slot))
    # optimizer_scope = tf.get_collection(tf.GraphKeys.VARIABLES, "opt")
    # optimizer_scope = [v for v in optimizer_scope if 'beta' not in v.name]
    # print(optimizer_scope)
    #
    # raise SystemExit

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        for i in range(args.epochs*1000):
            batch_x, _ = mnist.train.next_batch(args.batch_size)
            _, train_summ = sess.run([train_step, train_merged],
                                     feed_dict={x: gan.GAN.preprocess(batch_x)})

            # if i % 2 == 0:
            #     sess.run(gen_step,feed_dict={x: gan.GAN.preprocess(batch_x)})

            if i % 10 == 0:
                writer.add_summary(train_summ, i)

            if i % 100 == 0:
                L, test_summ = sess.run([gen_loss, test_merged],
                                                    feed_dict={x:
                            gan.GAN.preprocess(mnist.test.images[:100, ...])})
                print('\rStep: {} Loss: {}'.format(i, L), end='', flush=True)
                writer.add_summary(test_summ, i)

            if i % 500 == 0:
                # sess.run(tf.initialize_variables(optimizer_scope))
                save_path = saver.save(sess, os.path.join(args.logdir,"gan.ckpt"))
                print(save_path)

if __name__ == '__main__':
    main(argumentparser())
