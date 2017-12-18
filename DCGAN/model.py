import os
import datetime

import numpy as np
import tensorflow as tf

from ops import *
from utils import write_visit_input

class Generator:
    """Generate energy grid"""
    def __init__(self, *,
            batch_size,
            z_size,
            voxel_size=64,
            bottom_size=4,
            bottom_filters=256,
            name="generator",
            reuse=False):
        """
        Args:
            filter_unit: unit of minimal filter. n'th layer has the filter size
             of (filter_unit * 2**(n-1)).
        """
        self.batch_size = batch_size
        self.z_size = z_size
        self.voxel_size = voxel_size
        self.bottom_size = bottom_size
        self.bottom_filters = bottom_filters

        with tf.variable_scope(name, reuse=reuse):
            self._build()


    def _build(self):
        names = ("conv3d_trans_batchnorm_{}".format(i) for i in range(1000))

        self.training = tf.placeholder_with_default(
                            False, shape=(), name="training")

        self.z = tf.placeholder(
            shape=[self.batch_size, self.z_size], dtype=tf.float32)

        filters = self.bottom_filters

        bs = self.bottom_size
        with tf.variable_scope("bottom"):
            x = dense(self.z, units=bs*bs*bs*filters)
            x = tf.reshape(x, [self.batch_size, bs, bs, bs, filters])
            x = batch_normalization(x, training=self.training)
            x = tf.nn.relu(x)

        size = self.bottom_size

        while size < self.voxel_size:
            filters //= 2

            with tf.variable_scope(next(names)):
                x = conv3d_transpose(x, filters=filters)
                x = batch_normalization(x, training=self.training)
                x = tf.nn.relu(x)

            _, _, _, size, filters = x.get_shape().as_list()

        x = conv3d_transpose(
                x, filters=1, strides=1, use_bias=True,
                activation=tf.nn.sigmoid, name="outputs"
            )

        self.outputs = x


class Discriminator:
    """Calculate the probability that given grid data is an energy grid."""
    def __init__(self, *,
            x,
            batch_size,
            voxel_size,
            reuse=False,
            rate=0.5,
            top_size=4,
            filter_unit=32,
            minibatch=False,
            minibatch_kernel_size=1000,
            minibatch_dim_per_kernel=4,
            name="discriminator"):
        self.x = x
        self.batch_size = batch_size
        self.voxel_size = voxel_size
        self.rate = rate
        self.top_size = top_size
        self.filter_unit = filter_unit
        self.minibatch = minibatch
        self.minibatch_kernel_size = minibatch_kernel_size
        self.minibatch_dim_per_kernel = minibatch_dim_per_kernel

        with tf.variable_scope(name, reuse=reuse):
            self._build()


    def _build(self):
        names = ("conv3d_batchnorm_{}".format(i) for i in range(1000))

        self.training = tf.placeholder_with_default(
                            False, shape=(), name="training")

        filters = self.filter_unit

        with tf.variable_scope("bottom"):
            x = self.x
            x = tf.layers.dropout(x, rate=self.rate, training=self.training)
            x = conv3d(x, filters=filters, use_bias=1, strides=1)
            x = tf.nn.leaky_relu(x)

        size = self.voxel_size
        while self.top_size < size:
            filters *= 2

            with tf.variable_scope(next(names)):
                x = conv3d(x, filters=filters)
                x = batch_normalization(x, training=self.training)
                x = tf.nn.leaky_relu(x)

                _, _, _, size, filters = x.get_shape().as_list()

        x = tf.layers.flatten(x)

        if self.minibatch:
            x = minibatch_discrimination(
                    x,
                    num_kernels=self.minibatch_kernel_size,
                    dim_per_kernel=self.minibatch_dim_per_kernel,
                )

        self.logits = dense(x, units=1, name="logits", use_bias=True)
        self.outputs = tf.nn.sigmoid(self.logits, name="outputs")


class DCGAN:
    def __init__(self, *,
            dataset,
            logdir="logdir",
            batch_size,
            z_size,
            output_writer,
            save_every=1000,
            voxel_size=64,
            bottom_size=4,
            bottom_filters=256,
            rate=0.5,
            top_size=4,
            filter_unit=32,
            minibatch=False,
            minibatch_kernel_size=1000,
            minibatch_dim_per_kernel=4,
            l2_loss=False,
            g_learning_rate=0.0002,
            d_learning_rate=0.0002,
            train_gen_per_disc=1):

        try:
            os.makedirs(logdir)
        except Exception as e:
            print(e)

        self.date = datetime.datetime.now().isoformat()

        with open("{}/config-{}".format(logdir, self.date), "w") as f:
            f.write("""
                batch_size {}
                z_size {}
                voxel_size {}
                bottom_size {}
                bottom_filters {}
                rate {}
                top_size {}
                filter_unit {}
                minibatch {}
                minibatch_kernel_size {}
                minibatch_dim_per_kernel {}
                l2_loss {}
                g_learning_rate {}
                d_learning_rate {}
                train_gen_per_disc {}
                """.format(
                batch_size,
                z_size,
                voxel_size,
                bottom_size,
                bottom_filters,
                rate,
                top_size,
                filter_unit,
                minibatch,
                minibatch_kernel_size,
                minibatch_dim_per_kernel,
                l2_loss,
                g_learning_rate,
                d_learning_rate,
                train_gen_per_disc)
            )

        self.save_every = save_every
        self.output_writer = output_writer
        self.logdir = logdir
        self.batch_size = batch_size
        self.size = voxel_size
        self.train_gen_per_disc = train_gen_per_disc
        # Make iterator from the dataset.
        self.iterator = dataset.batch(batch_size).make_initializable_iterator()

        # Build nueral network.
        self.generator = Generator(
            batch_size=batch_size,
            z_size=z_size,
            voxel_size=voxel_size,
            bottom_size=bottom_size,
            bottom_filters=bottom_filters,
        )

        self.discriminator_real = Discriminator(
            x=self.iterator.get_next(), # dataset.
            batch_size=batch_size,
            voxel_size=voxel_size,
            rate=rate,
            top_size=top_size,
            filter_unit=filter_unit,
            minibatch=minibatch,
            minibatch_kernel_size=minibatch_kernel_size,
            minibatch_dim_per_kernel=minibatch_dim_per_kernel,
        )

        self.discriminator_fake = Discriminator(
            x=self.generator.outputs,
            batch_size=batch_size,
            voxel_size=voxel_size,
            rate=rate,
            top_size=top_size,
            filter_unit=filter_unit,
            minibatch=minibatch,
            minibatch_kernel_size=minibatch_kernel_size,
            minibatch_dim_per_kernel=minibatch_dim_per_kernel,
            reuse=True,
        )

        # Build losses.
        train_vars = tf.trainable_variables()

        d_vars = [v for v in train_vars if v.name.startswith("discriminator/")]
        g_vars = [v for v in train_vars if v.name.startswith("generator/")]

        with tf.variable_scope("loss/real"):
            real_logits = self.discriminator_real.logits
            real_loss = -tf.reduce_mean(
                sigmoid_log_with_logits(real_logits)
            )

        with tf.variable_scope("loss/fake"):
            fake_logits = self.discriminator_fake.logits
            fake_loss = -tf.reduce_mean(
                sigmoid_log_with_logits(-fake_logits)
            )

        with tf.variable_scope("loss/disc"):
            d_loss = real_loss + fake_loss

        with tf.variable_scope("loss/gen"):
            g_loss = -tf.reduce_mean(
                sigmoid_log_with_logits(fake_logits)
            )

        # Build train ops.
        with tf.variable_scope("train/disc"):
            d_optimizer = tf.train.AdamOptimizer(
                              learning_rate=d_learning_rate, beta1=0.5)

            self.d_train_op = d_optimizer.minimize(d_loss, var_list=d_vars)

        with tf.variable_scope("train/gen"):
            g_optimizer = tf.train.AdamOptimizer(
                              learning_rate=g_learning_rate, beta1=0.5)

            self.g_train_op = g_optimizer.minimize(g_loss, var_list=g_vars)

        # Build vars_to_save.
        moving_avg_vars = tf.moving_average_variables()
        self.vars_to_save = d_vars + g_vars + moving_avg_vars

        # Build summaries.
        with tf.name_scope("scalar_summary"):
            tf.summary.scalar("disc", d_loss)
            tf.summary.scalar("gen", g_loss)
            tf.summary.scalar("real", real_loss)
            tf.summary.scalar("fake", fake_loss)

        with tf.name_scope("histogram_summary"):
            for v in self.vars_to_save:
                tf.summary.histogram(v.name, v)

        self.merged_summary = tf.summary.merge_all()


    def train(self):
        # Make log paths.
        logdir = self.logdir

        date = self.date

        writer_name = "{}/run-{}".format(logdir, date)
        saver_name = "{}/save-{}".format(logdir, date)
        sample_dir = "{}/samples-{}".format(logdir, date)

        # Make directory
        try:
            os.makedirs(sample_dir)
        except:
            print("error on os.mkdir?")

        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)
        file_writer = tf.summary.FileWriter(
                          writer_name, tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)

            for i in range(100000000):
                z = np.random.uniform(-1.0, 1.0,
                        size=[self.batch_size, self.generator.z_size])

                # Train discriminator.
                feed_dict = {
                    self.generator.z: z,
                    self.generator.training: True,
                    self.discriminator_real.training: True,
                    self.discriminator_fake.training: True,
                }

                sess.run([self.d_train_op], feed_dict=feed_dict)

                for _ in range(self.train_gen_per_disc):
                    sess.run([self.g_train_op], feed_dict=feed_dict)

                if i % self.save_every == 0:
                    feed_dict = {
                        self.generator.z: z,
                    }

                    run_options = tf.RunOptions(
                                      trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    summary_str, samples = sess.run(
                       [self.merged_summary, self.generator.outputs],
                       feed_dict=feed_dict,
                       options=run_options,
                       run_metadata=run_metadata,
                    )

                    file_writer.add_run_metadata(
                        run_metadata, "step_{}".format(i)
                    )
                    file_writer.add_summary(summary_str, i)

                    saver.save(sess, saver_name, global_step=i)

                    # Generate energy grid samples
                    for j, sample in enumerate(samples):
                        stem = "sample_{}".format(j)
                        self.output_writer(stem, sample, self.size, sample_dir)

                print("{}/{}  ITER: {}".format(logdir, date, i))


    def generate_samples(self, checkpoint, n_samples=5):
        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)

        j = 0
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            try:
                os.makedirs("samples")
            except Exception as e:
                print(e)
                #return

            size = self.batch_size

            z1 = np.random.uniform(-1.0, 1.0,
                        size=[self.generator.z_size])

            for i in range(n_samples):
                print(i)
                z0 = z1

                z1 = np.random.uniform(-1.0, 1.0,
                            size=[self.generator.z_size])

                x_space = np.linspace(0, 1, size, endpoint=True)

                z = [(1.0-x)*z0 + x*z1 for x in x_space]
                z = np.array(z)

                feed_dict = {
                    self.generator.z: z,
                }

                samples = sess.run(
                    self.generator.outputs,
                    feed_dict=feed_dict,
                )

                for j, sample in enumerate(samples, start=j+1):
                    stem = "sample_{}".format(j)
                    write_visit_input(
                        stem, sample, self.size, "samples",
                        energy_range=[-5000, 5000])
