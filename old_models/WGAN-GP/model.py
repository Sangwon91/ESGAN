import os
import math
import glob
import datetime

import numpy as np
import tensorflow as tf

from ops import *

class Generator:
    """Generate energy grid"""
    def __init__(self, *,
            batch_size,
            z_size,
            voxel_size,
            bottom_size,
            bottom_filters,
            name="generator",
            reuse=False,
            z=None):
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
        self.z = z

        with tf.variable_scope(name, reuse=reuse):
            self._build()


    def _build(self):
        names = ("conv3d_trans_batchnorm_{}".format(i) for i in range(1000))

        self.training = tf.placeholder_with_default(
                            False, shape=(), name="training")

        # Create z if not feeded.
        if self.z is None:
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

        x = conv3d_transpose(x,
                filters=1,
                strides=1,
                use_bias=True,
                bias_initializer=tf.constant_initializer(0.5),
                activation=tf.nn.sigmoid,
                name="outputs",
            )

        self.outputs = x


class Discriminator:
    """Calculate the probability that given grid data is an energy grid."""
    def __init__(self, *,
            x,
            batch_size,
            voxel_size,
            rate,
            top_size,
            filter_unit,
            minibatch,
            minibatch_kernel_size,
            minibatch_dim_per_kernel,
            reuse=False,
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
                #x = batch_normalization(x, training=self.training)
                # Layer normalization. alternative of batchnorm
                x = tf.contrib.layers.layer_norm(x)
                x = tf.nn.leaky_relu(x)

                _, _, _, size, filters = x.get_shape().as_list()

        x = tf.layers.flatten(x)

        if self.minibatch:
            x = minibatch_discrimination(
                    x,
                    num_kernels=self.minibatch_kernel_size,
                    dim_per_kernel=self.minibatch_dim_per_kernel,
                )

        self.logits = dense(x,
                          units=1,
                          use_bias=True,
                          name="logits",
                      )

        # No output activation for WGAN.
        self.outputs = self.logits
        #self.outputs = tf.nn.sigmoid(self.logits, name="outputs")


class WGANGP:
    def __init__(self, *,
            dataset,
            logdir,
            batch_size,
            z_size,
            output_writer,
            save_every,
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
            n_critics,
            gp_lambda):

        try:
            os.makedirs(logdir)
        except Exception as e:
            print(e)

        self.date = datetime.datetime.now().isoformat()

        self.save_every = save_every
        self.output_writer = output_writer
        self.logdir = logdir
        self.batch_size = batch_size
        self.size = voxel_size
        self.n_critics = n_critics
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

        # Make interpolated inputs
        real_data = self.discriminator_real.x
        fake_data = self.generator.outputs

        eps = tf.random_uniform(
                  [self.batch_size, 1, 1, 1, 1], minval=0.0, maxval=1.0)

        interps = eps*(fake_data-real_data) + real_data

        self.discriminator_interp = Discriminator(
            x=interps,
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

        with tf.variable_scope("loss/gradient"):
            gradients, = tf.gradients(
                             self.discriminator_interp.outputs,
                             [interps],
                         )

            gradients = tf.layers.flatten(gradients)
            norms = tf.sqrt(tf.reduce_sum(gradients**2, axis=1))
            gradient_penalty = tf.reduce_mean((norms-1.0)**2)

        with tf.variable_scope("loss/real"):
            real_logits = self.discriminator_real.logits
            real_loss = tf.reduce_mean(real_logits)

        with tf.variable_scope("loss/fake"):
            fake_logits = self.discriminator_fake.logits
            fake_loss = tf.reduce_mean(fake_logits)

        with tf.variable_scope("loss/disc"):
            d_loss = fake_loss - real_loss + gp_lambda*gradient_penalty

        with tf.variable_scope("loss/gen"):
            g_loss = -fake_loss

        # Build train ops.
        with tf.variable_scope("train/disc"):
            d_optimizer = tf.train.AdamOptimizer(
                              learning_rate=d_learning_rate,
                              beta1=0.5,
                              beta2=0.9,
                          )

            self.d_train_op = d_optimizer.minimize(d_loss, var_list=d_vars)

        with tf.variable_scope("train/gen"):
            g_optimizer = tf.train.AdamOptimizer(
                              learning_rate=g_learning_rate,
                              beta1=0.5,
                              beta2=0.9
                          )

            self.g_train_op = g_optimizer.minimize(g_loss, var_list=g_vars)

        # Build vars_to_save.
        moving_avg_vars = tf.moving_average_variables()
        self.vars_to_save = d_vars + g_vars + moving_avg_vars

        # Build summaries.
        with tf.name_scope("tensorboard"):
            tf.summary.scalar("loss/disc", d_loss)
            tf.summary.scalar("loss/gen", g_loss)
            tf.summary.scalar("loss/real", real_loss)
            tf.summary.scalar("loss/fake", fake_loss)
            tf.summary.scalar("loss/grad", gradient_penalty)
            # To see convergence.
            tf.summary.scalar("loss/negative_critic", real_loss-fake_loss)

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
                    self.discriminator_real.training: True,   # Meaningless.
                    self.discriminator_fake.training: True,   # Meaningless.
                    self.discriminator_interp.training: True, # Meaningless.
                }
                for _ in range(self.n_critics):
                    sess.run([self.d_train_op], feed_dict=feed_dict)

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
                        self.output_writer(
                            stem=stem,
                            grid=sample,
                            size=self.size,
                            save_dir=sample_dir)

                print("{}/{}  ITER: {}".format(logdir, date, i))


    def generate_samples(self, sample_dir, checkpoint, n_samples):
        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            try:
                os.makedirs(sample_dir)
            except Exception as e:
                print(e)

            size = self.batch_size

            idx = 0
            n_iters = math.ceil(n_samples / size)
            for i in range(n_iters):
                print("... Generating {:02d}%".format((100*i)//n_iters))

                z = np.random.uniform(-1.0, 1.0,
                        size=[size, self.generator.z_size])

                feed_dict = {
                    self.generator.z: z,
                }

                samples = sess.run(
                    self.generator.outputs,
                    feed_dict=feed_dict,
                )

                # Generate energy grid samples
                for sample in samples:
                    name = "{}/ann_{}.griddata".format(sample_dir, idx)
                    #self.output_writer(stem, sample, self.size,
                    #    save_dir=sample_dir)
                    sample = 1.0 - sample
                    sample = (5000.0 - (-3000.0))*sample + (-3000.0)
                    sample = sample.astype(np.float32)
                    sample.tofile(name)

                    idx += 1

            print("Done")


    def interpolate_samples(self, sample_dir, checkpoint, n_samples):
        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            try:
                os.makedirs(sample_dir)
            except Exception as e:
                print(e)

            size = self.batch_size
            z_size = self.generator.z_size

            idx = 0
            n_iters = n_samples

            interval = np.linspace(0, 1, size)
            z0 = np.random.uniform(-1.0, 1.0, size=[z_size])

            for i in range(n_iters):
                print("... Generating {:02d}%".format((100*i)//n_iters))

                z1 = np.random.uniform(-1.0, 1.0, size=[z_size])
                z = np.array([(1-t)*z0 + t*z1 for t in interval])

                feed_dict = {
                    self.generator.z: z,
                }

                samples = sess.run(
                    self.generator.outputs,
                    feed_dict=feed_dict,
                )

                # Generate energy grid samples
                for sample in samples:
                    name = "{}/ann_{}.griddata".format(sample_dir, idx)
                    #self.output_writer(stem, sample, self.size,
                    #    save_dir=sample_dir)
                    sample = 1.0 - sample
                    sample = (5000.0 - (-3000.0))*sample + (-3000.0)
                    sample = sample.astype(np.float32)
                    sample.tofile(name)

                    idx += 1

                z0 = z1

            print("Done")
