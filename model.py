import os
import math
import glob
import datetime
import platform
import itertools

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

        # Create z if not fed.
        if self.z is None:
            self.z = tf.random_normal(
                         shape=[self.batch_size, self.z_size],
                         mean=0.0,
                         stddev=1.0,
                         dtype=tf.float32,
                         name="z",
                     )

        z = self.z

        filters = self.bottom_filters
        bs = self.bottom_size
        with tf.variable_scope("bottom"):
            x = dense(z, units=bs*bs*bs*filters)
            x = tf.reshape(x, [self.batch_size, bs, bs, bs, filters])
            x = batch_normalization(x, training=self.training)
            x = tf.nn.relu(x)
            saved_x = x

        with tf.variable_scope("generate_cell"):
            # Single hidden layer.
            x = tf.layers.flatten(x)
            x = dense(x, units=512)
            x = batch_normalization(
                    x, training=self.training, global_norm=False)
            x = tf.nn.relu(x)
            # Final layer, no batchnorm.
            x = dense(x, units=3, use_bias=True, name="c_logits")

            self.c_logits = x
            self.c_outputs = tf.nn.sigmoid(x, name="c_outputs")

        # Restore bottom end.
        x = saved_x
        size = self.bottom_size
        while size < self.voxel_size:
            filters //= 2

            with tf.variable_scope(next(names)):
                x = conv3d_transpose(x, filters=filters)
                x = batch_normalization(x, training=self.training)
                x = tf.nn.relu(x)

            _, _, _, size, filters = x.get_shape().as_list()

        with tf.variable_scope("outputs"):
            x = conv3d_transpose(x,
                    filters=1,
                    strides=1,
                    use_bias=True,
                    bias_initializer=tf.constant_initializer(0.5),
                    activation=tf.nn.sigmoid,
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
            x = conv3d(x, filters=filters, use_bias=True, strides=1)
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
        x_before_minibatch = x

        if self.minibatch:
            x = minibatch_discrimination(
                    x,
                    num_kernels=self.minibatch_kernel_size,
                    dim_per_kernel=self.minibatch_dim_per_kernel,
                )

        self.logits = dense(x,
                          units=1,
                          name="logits",
                      )
        self.outputs = tf.nn.sigmoid(self.logits, name="outputs")

        with tf.variable_scope("cell_inference"):
            # Single hidden layer.
            x = dense(x_before_minibatch, units=512)
            x = batch_normalization(
                    x, training=self.training, global_norm=False)
            x = tf.nn.relu(x)
            # Final layer, no batchnorm.
            x = dense(x, units=3, use_bias=True, name="c_logits")

            self.c_logits = x
            self.c_outputs = tf.nn.sigmoid(x, name="c_outputs")


class ESGAN:
    def __init__(self, *,
            dataset,
            logdir,
            batch_size,
            z_size,
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
            train_gen_per_disc,
            in_temper,
            feature_matching):

        try:
            os.makedirs(logdir)
        except Exception as e:
            print(e)

        self.date = datetime.datetime.now().isoformat()

        self.save_every = save_every
        self.logdir = logdir
        self.batch_size = batch_size
        self.size = voxel_size
        self.train_gen_per_disc = train_gen_per_disc
        self.in_temper = in_temper
        self.feature_matching = feature_matching
        self.dataset = dataset
        # Make iterator from the dataset.
        with tf.variable_scope("build_dataset"):
            # Make iterator from the dataset.
            self.iterator = (
                self.dataset.dataset
                .batch(batch_size)
                .make_initializable_iterator()
            )

            # cell_data, grid_data = next_data.
            self.next_data = self.iterator.get_next()

        # Build nueral network.
        self.generator = Generator(
            batch_size=batch_size,
            z_size=z_size,
            voxel_size=voxel_size,
            bottom_size=bottom_size,
            bottom_filters=bottom_filters,
        )

        self.discriminator_real = Discriminator(
            x=self.next_data[1], # Takes grid only.
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

        # Loss function parameters.
        one_side_label = 0.8
        weight_cell_real = 1.0
        weight_cell_fake = 0.1

        with tf.variable_scope("loss/real"):
            real_logits = self.discriminator_real.logits
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=real_logits,
                    labels=one_side_label*tf.ones_like(real_logits),
                )
            )

        with tf.variable_scope("loss/fake"):
            fake_logits = self.discriminator_fake.logits
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_logits,
                    labels=tf.zeros_like(fake_logits),
                )
            )

        with tf.variable_scope("loss/real_cell"):
            real_c_logits = self.discriminator_real.c_logits
            real_c_outputs = self.discriminator_real.c_outputs

            real_c_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                              labels=self.next_data[0],
                              logits=real_c_logits,
                          )
            real_c_loss = tf.reduce_mean(real_c_loss)

            real_c_abs_diff = tf.abs(real_c_outputs-self.next_data[0])
            real_c_abs_diff = tf.reduce_mean(real_c_abs_diff)

        with tf.variable_scope("loss/fake_cell"):
            fake_c_logits = self.discriminator_fake.c_logits
            fake_c_outputs = self.discriminator_fake.c_outputs

            fake_c_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                              labels=fake_c_outputs,
                              logits=self.generator.c_logits,
                          )
            fake_c_loss = tf.reduce_mean(fake_c_loss)

            fake_c_abs_diff = tf.abs(fake_c_outputs-self.generator.c_outputs)
            fake_c_abs_diff = tf.reduce_mean(fake_c_abs_diff)

        with tf.variable_scope("loss/disc"):
            d_loss = real_loss + fake_loss
            d_total_loss = d_loss + weight_cell_real*real_c_loss

        with tf.variable_scope("loss/feature_matching"):
            # MODIFIED. (Response of "Warning. It's hard-corded.")
            lower, upper = self.dataset.energy_scale

            real_x = self.discriminator_real.x
            if self.dataset.invert:
                real_x = 1.0 - real_x
            real_x = (upper - lower)*real_x + lower

            fake_x = self.discriminator_fake.x
            if self.dataset.invert:
                fake_x = 1.0 - fake_x
            fake_x = (upper - lower)*fake_x + lower

            default_temper = 300.0
            self.temper = tf.placeholder_with_default(
                              default_temper,
                              shape=[],
                              name="temperature",
                          )

            temper = self.temper

            # Chemical potentials.
            # Save variables for learning of generator.
            saved_real_mean = tf.Variable(
                0.0, dtype=tf.float32, trainable=False, name="saved_real_mean")
            saved_real_std = tf.Variable(
                1.0, dtype=tf.float32, trainable=False, name="saved_real_std")

            real_boltz = tf.exp(-real_x / temper)
            real_boltz = tf.reduce_mean(real_boltz, axis=[1,2,3,4])
            real_cp = tf.log(real_boltz)
            # Calculate statistics of the reals.
            real_mean, real_std = tf.nn.moments(real_cp, axes=[0])
            real_std = tf.sqrt(real_std)

            # save ops
            with tf.variable_scope("fm_save_ops"):
                save_real_ops = [
                    tf.assign(saved_real_mean, real_mean),
                    tf.assign(saved_real_std, real_std),
                ]

            fake_boltz = tf.exp(-fake_x / temper)
            fake_boltz = tf.reduce_mean(fake_boltz, axis=[1,2,3,4])
            fake_cp = tf.log(fake_boltz)
            # Calculate statistics of the fakes.
            fake_mean, fake_std = tf.nn.moments(fake_cp, axes=[0])
            fake_std = tf.sqrt(fake_std)

            # Use saved mean and std.
            fm_loss = (
                tf.abs(saved_real_mean - fake_mean) +
                tf.abs(saved_real_std - fake_std)
            )

        with tf.variable_scope("loss/gen"):
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_logits,
                    labels=tf.ones_like(fake_logits),
                )
            )

            g_total_loss = g_loss + weight_cell_fake*fake_c_loss

            if self.feature_matching:
                g_total_loss += fm_loss

        # Build train ops.
        with tf.variable_scope("train/disc"):
            d_optimizer = tf.train.AdamOptimizer(
                              learning_rate=d_learning_rate, beta1=0.5)

            with tf.control_dependencies(save_real_ops):
                self.d_train_op = d_optimizer.minimize(
                                      d_total_loss,
                                      var_list=d_vars,
                                  )

        with tf.variable_scope("train/gen"):
            g_optimizer = tf.train.AdamOptimizer(
                              learning_rate=g_learning_rate, beta1=0.5)

            self.g_train_op = g_optimizer.minimize(
                                  g_total_loss,
                                  var_list=g_vars,
                              )

        # Build vars_to_save.
        moving_avg_vars = tf.moving_average_variables()
        self.vars_to_save = d_vars + g_vars + moving_avg_vars

        # Build summaries.
        g_summaries = list()
        d_summaries = list()

        with tf.name_scope("scalar_summaries"):
            d_summaries += [
                tf.summary.scalar("d_total_loss", d_total_loss),
                tf.summary.scalar("d_gan_loss", d_loss),
                tf.summary.scalar("d_real_loss", real_loss),
                tf.summary.scalar("d_fake_loss", fake_loss),
                tf.summary.scalar("d_real_c_loss", real_c_loss),
                tf.summary.scalar("d_fake_c_loss", fake_c_loss),
                tf.summary.scalar("d_real_c_abs_diff", real_c_abs_diff),
                tf.summary.scalar("d_fake_c_abs_diff", fake_c_abs_diff),
            ]

            g_summaries += [
                tf.summary.scalar("g_total_loss", g_total_loss),
                tf.summary.scalar("g_gan_loss", g_loss),
                tf.summary.scalar("g_feature_matching_loss", fm_loss),
                tf.summary.scalar("g_temperature", self.temper),
                tf.summary.scalar("g_fake_c_loss", fake_c_loss),
                tf.summary.scalar("g_fake_c_abs_diff", fake_c_abs_diff),
            ]

        with tf.name_scope("histogram_summaries"):
            d_moving_vars = [
                v for v in moving_avg_vars if v.name.startswith("d")
            ]
            g_moving_vars = [
                v for v in moving_avg_vars if v.name.startswith("g")
            ]

            d_summaries += [
                tf.summary.histogram(v.name, v) for v in d_vars+d_moving_vars
            ]

            g_summaries += [
                tf.summary.histogram(v.name, v) for v in g_vars+g_moving_vars
            ]


        with tf.name_scope("output_histogram_summaries"):
            gen_c = self.generator.c_outputs

            d_summaries += [
                tf.summary.histogram("d_real_c_infer", real_c_outputs),
                tf.summary.histogram("d_real_c_input", self.next_data[0]),
                tf.summary.histogram("d_fake_c_infer", fake_c_outputs),
                tf.summary.histogram("d_fake_c_input", gen_c),
            ]

            g_summaries += [
                tf.summary.histogram("g_fake_c_infer", fake_c_outputs),
                tf.summary.histogram("g_fake_c_input", gen_c),
            ]

        self.d_merged_summary = tf.summary.merge(d_summaries)
        self.g_merged_summary = tf.summary.merge(g_summaries)

    def train(self, checkpoint=None, start_step=0):
        # Process Informations.
        node = platform.uname()[1]
        pid = os.getpid()
        # Make log paths.
        logdir = self.logdir
        # Short alias for date.
        date = self.date

        writer_name = "{}/run-{}".format(logdir, date)
        saver_name = "{}/save-{}".format(logdir, date)
        sample_dir = "{}/samples-{}".format(logdir, date)

        # Make directory
        try:
            os.makedirs(sample_dir)
        except:
            print("error on os.mkdir?")

        # max_to_keep=None will save all checkpoints.
        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=None)
        file_writer = tf.summary.FileWriter(
                          writer_name, tf.get_default_graph())

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)

            if checkpoint:
                print("Restoring:", checkpoint)
                saver.restore(sess, checkpoint)

            for i in itertools.count(start=start_step):
                t1 = datetime.datetime.now()

                # Train discriminator.
                feed_dict = {
                    self.temper: self.in_temper,
                    self.generator.training: True,
                    self.discriminator_real.training: True,
                    self.discriminator_fake.training: True,
                }

                save_step = (i%self.save_every == 0)

                # Build fetches.
                d_fetches = [self.d_merged_summary, self.d_train_op]
                g_fetches = [
                    self.g_merged_summary,
                    self.generator.c_outputs,
                    self.generator.outputs,
                    self.g_train_op,
                ]

                if not save_step:
                    # Take train ops only.
                    d_fetches = d_fetches[-1]
                    g_fetches = g_fetches[-1]

                d_result = sess.run(d_fetches, feed_dict=feed_dict)

                for _ in range(self.train_gen_per_disc):
                    g_result = sess.run(g_fetches, feed_dict=feed_dict)

                if save_step:
                    d_summary_str, _ = d_result
                    g_summary_str, cells, grids, _ = g_result

                    file_writer.add_summary(d_summary_str, i)
                    file_writer.add_summary(g_summary_str, i)

                    saver.save(sess, saver_name, global_step=i)

                    for j, (cell, grid) in enumerate(zip(cells, grids)):
                        stem = "sample_{}".format(j)
                        self.dataset.write_visit_sample(
                            cell=cell,
                            grid=grid,
                            stem=stem,
                            save_dir=sample_dir,
                        )

                # Calculate duration time of a step.
                t2 = datetime.datetime.now()
                dt = t2 - t1
                duration = "{:.3f}".format(dt.seconds + dt.microseconds*1e-6)

                print(
                    "NODE:{}, PID:{}, NAME:{}/{}, ITER: {}, DURATION: {} sec"
                    .format(node, pid, logdir, date, i, duration)
                )

    def generate_samples(self, *, sample_dir, checkpoint, n_samples):
        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            try:
                print("Try to make save directory...")
                os.makedirs(sample_dir)
            except Exception as e:
                print(e)
                print("Stop generation")
                return

            size = self.batch_size

            idx = 0
            n_iters = math.ceil(n_samples / size)
            for i in range(n_iters):
                print("... Generating {:d}".format(idx))

                fetches = [
                    self.generator.c_outputs,
                    self.generator.outputs,
                ]

                feed_dict = {}

                cells, samples = sess.run(
                    fetches=fetches,
                    feed_dict=feed_dict,
                )

                # Generate energy grid samples
                for cell, sample in zip(cells, samples):
                    stem = "ann_{}".format(idx)
                    self.dataset.write_sample(
                        cell=cell,
                        grid=sample,
                        stem=stem,
                        save_dir=sample_dir,
                    )

                    idx += 1

            print("Done")

    def interpolate_samples(self, *, sample_dir, checkpoint, n_samples):
        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            try:
                print("Try to make save directory...")
                os.makedirs(sample_dir)
            except Exception as e:
                print(e)
                print("Stop generation")
                return

            size = self.batch_size
            z_size = self.generator.z_size

            idx = 0
            n_iters = n_samples

            thetas = np.linspace(0, 0.5*np.pi, size, endpoint=False)

            z0 = np.random.normal(0.0, 1.0, size=[z_size])
            z_init = np.array(z0) # Copy start z.
            for i in range(n_iters):
                print("... Generating {:d}".format(idx))

                if i == n_iters-1:
                    z1 = z_init
                else:
                    z1 = np.random.normal(0.0, 1.0, size=[z_size])

                z = np.array(
                    [(math.cos(t)*z0 + math.sin(t)*z1) for t in thetas]
                )

                fetches = [
                    self.generator.c_outputs,
                    self.generator.outputs,
                ]

                feed_dict = {
                    self.generator.z: z,
                }

                cells, samples = sess.run(
                    fetches=fetches,
                    feed_dict=feed_dict,
                )

                # Generate energy grid samples
                for cell, sample in zip(cells, samples):
                    stem = "ann_{}".format(idx)
                    self.dataset.write_sample(
                        cell=cell,
                        grid=sample,
                        stem=stem,
                        save_dir=sample_dir,
                    )

                    idx += 1

                z0 = z1

            print("Done")

    def generate_sample_from_fixed_z(self, *, z, sample_dir, checkpoint):
        if self.batch_size != 1:
            raise Exception("batch_size != 1")

        if z.size != self.generator.z_size:
            raise Exception("difference z_size is fed.")
        # Same as expand_dims but more general.
        z = z.reshape([1, z.size])

        # Extract simulation step from ckpt.
        step = checkpoint.split("-")[-1]

        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)

            try:
                print("Try to make save directory...")
                os.makedirs(sample_dir)
            except Exception as e:
                print(e)
                print("Keep generation")

            size = self.batch_size

            fetches = [
                self.generator.c_outputs,
                self.generator.outputs,
            ]

            feed_dict = {
                self.generator.z: z,
            }

            cells, samples = sess.run(
                fetches=fetches,
                feed_dict=feed_dict,
            )

            # Generate energy grid samples
            cell, sample = (cells[0, ...], samples[0, ...])

            stem = "ann_{}".format(step)
            self.dataset.write_sample(
                cell=cell,
                grid=sample,
                stem=stem,
                save_dir=sample_dir,
            )

            print("Done,", step)
