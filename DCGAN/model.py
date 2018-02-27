import os
import math
import glob
import datetime
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

        # Create z if not feeded.
        if self.z is None:
            self.z = tf.random_normal(
                         shape=[self.batch_size, self.z_size],
                         mean=0.0,
                         stddev=1.0,
                         dtype=tf.float32,
                         name="z",
                     )

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

        with tf.variable_scope("outputs"):
            x = conv3d_transpose(x,
                    filters=1,
                    strides=1,
                    use_bias=True,
                    bias_initializer=tf.constant_initializer(0.5),
                    activation=tf.nn.sigmoid,
                    #name="outputs",
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

        self.logits = dense(x,
                          units=1,
                          #use_bias=True,
                          name="logits",
                      )
        self.outputs = tf.nn.sigmoid(self.logits, name="outputs")


class DCGAN:
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
            train_gen_per_disc):

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
        self.dataset = dataset
        # Make iterator from the dataset.
        self.iterator = (
            self.dataset.dataset
            .batch(batch_size)
            .make_initializable_iterator()
        )

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
            real_boltz = tf.exp(-real_x / temper)
            real_boltz = tf.reduce_mean(real_boltz, axis=[1,2,3,4])
            real_cp = tf.log(real_boltz)
            real_avg_cp = tf.reduce_mean(real_cp)

            fake_boltz = tf.exp(-fake_x / temper)
            fake_boltz = tf.reduce_mean(fake_boltz, axis=[1,2,3,4])
            fake_cp = tf.log(fake_boltz)
            fake_avg_cp = tf.reduce_mean(fake_cp)

            #fm_loss = (real_avg_cp - fake_avg_cp)**2
            fm_loss = tf.abs(real_avg_cp - fake_avg_cp)

            self.feature_matching = tf.placeholder_with_default(
                                        True,
                                        shape=[],
                                        name="feature_matching"
                                    )

            g_total_loss = tf.cond(
                               self.feature_matching,
                               lambda: g_loss+fm_loss,
                               lambda: g_loss,
                               name="g_total_loss",
                           )

        # Build train ops.
        with tf.variable_scope("train/disc"):
            d_optimizer = tf.train.AdamOptimizer(
                              learning_rate=d_learning_rate, beta1=0.5)

            self.d_train_op = d_optimizer.minimize(
                                  d_loss,
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
        with tf.name_scope("scalar_summary"):
            tf.summary.scalar("disc", d_loss)
            tf.summary.scalar("gen", g_loss)
            tf.summary.scalar("real", real_loss)
            tf.summary.scalar("fake", fake_loss)
            tf.summary.scalar("feature_matching", fm_loss)
            tf.summary.scalar("temperature", self.temper)

        with tf.name_scope("histogram_summary"):
            for v in self.vars_to_save:
                tf.summary.histogram(v.name, v)

        self.merged_summary = tf.summary.merge_all()


    def train(self, checkpoint=None, start_step=0):
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

        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=2)
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
                # Train discriminator.
                feed_dict = {
                    #self.feature_matching: (i > 50000),
                    self.generator.training: True,
                    self.discriminator_real.training: True,
                    self.discriminator_fake.training: True,
                }

                sess.run([self.d_train_op], feed_dict=feed_dict)

                for _ in range(self.train_gen_per_disc):
                    sess.run([self.g_train_op], feed_dict=feed_dict)

                if i % self.save_every == 0:
                    feed_dict = {}

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
                        self.dataset.write_visit_sample(
                            x=sample,
                            stem=stem,
                            save_dir=sample_dir,
                        )

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
                print("... Generating {:d}".format(idx))

                feed_dict = {}

                samples = sess.run(
                    self.generator.outputs,
                    feed_dict=feed_dict,
                )

                # Generate energy grid samples
                for sample in samples:
                    stem = "ann_{}".format(idx)
                    self.dataset.write_sample(
                        x=sample,
                        stem=stem,
                        save_dir=sample_dir,
                    )

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

            thetas = np.linspace(0, 0.5*np.pi, size)

            z0 = np.random.uniform(-1.0, 1.0, size=[z_size])
            for i in range(n_iters):
                print("... Generating {:d}".format(idx))

                z1 = np.random.uniform(-1.0, 1.0, size=[z_size])
                z = np.array(
                    [(math.cos(t)*z0 + math.sin(t)*z1) for t in thetas]
                )

                feed_dict = {
                    self.generator.z: z,
                }

                samples = sess.run(
                    self.generator.outputs,
                    feed_dict=feed_dict,
                )

                # Generate energy grid samples
                for sample in samples:
                    stem = "ann_{}".format(idx)
                    self.dataset.write_sample(
                        x=sample,
                        stem=stem,
                        save_dir=sample_dir,
                    )

                    idx += 1

                z0 = z1

            print("Done")


class Frac2Cell:
    """Calculate cell parameters (a, b, c) from the fractional energy grid."""
    def __init__(self, *,
            dataset,
            validset,
            logdir,
            batch_size,
            voxel_size,
            rate,
            top_size,
            filter_unit,
            learning_rate,
            save_every,
            scale_factor,
            reuse=False,
            name="frac2cell"):
        """
        x: (cells, fractional_grids) tuple.
           Warning: Not a (cell, fractional_grid)s.
                    Please be aware of the difference.

           cell is normalized.
        """
        self.dataset = dataset
        self.validset = validset

        with tf.variable_scope("build_dataset"):
            with tf.variable_scope("training"):
                # Make iterator from the dataset.
                self.iterator = (
                    self.dataset.dataset
                    .batch(batch_size)
                    .make_initializable_iterator()
                )

                self.next_data = self.iterator.get_next()

            with tf.variable_scope("validation"):
                # Make iterator from the dataset.
                self.valid_iterator = (
                    self.validset.dataset
                    .batch(batch_size)
                    .make_initializable_iterator()
                )

                self.next_valid_data = self.valid_iterator.get_next()

        self.x = self.next_data
        self.logdir = logdir
        self.batch_size = batch_size
        self.voxel_size = voxel_size
        self.rate = rate
        self.top_size = top_size
        self.filter_unit = filter_unit
        self.learning_rate = learning_rate
        self.save_every = save_every
        self.scale_factor = scale_factor
        self.date = datetime.datetime.now().isoformat()

        with tf.variable_scope(name, reuse=reuse):
            self._build()


    def _build(self):
        names = ("conv3d_batchnorm_{}".format(i) for i in range(1000))
        ms_names = ("match_shape_{}".format(i) for i in range(1000))
        skip_name = ("skip_{}".format(i) for i in range(1000))

        def import_info(x, info):
            return x + info

        self.training = tf.placeholder_with_default(
                            False, shape=(), name="training")

        filters = self.filter_unit

        with tf.variable_scope("bottom"):
            x = self.x[1]
            x = tf.layers.dropout(x, rate=self.rate, training=self.training)
            x = conv3d(x, filters=filters, use_bias=True, strides=1)
            x = tf.nn.leaky_relu(x)

        # Save for skip connection.
        first_layer = x

        size = self.voxel_size
        while self.top_size < size:
            filters *= 2

            with tf.variable_scope(next(names)):
                x = conv3d(x, filters=filters)
                x = batch_normalization(x, training=self.training)

            x = tf.nn.leaky_relu(x)

            _, _, _, size, filters = x.get_shape().as_list()

            with tf.variable_scope(next(skip_name)):
                # Match shape
                y = conv3d(first_layer, filters=filters,
                        strides=self.voxel_size//size)
                y = batch_normalization(y, training=self.training)

            y = tf.nn.leaky_relu(y)

            x = x + y

            _, _, _, size, filters = x.get_shape().as_list()

        x = tf.layers.flatten(x)

        self.logits = dense(x, units=3, name="logits", use_bias=False)
        self.outputs = tf.nn.sigmoid(self.logits, name="outputs")

        cell = self.x[0]
        cell.set_shape([self.batch_size, 3])
        self.cell = cell

        with tf.variable_scope("loss"):
            cross_entopry = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=cell,
                logits=self.logits,
                name="cross_entopry",
            )
            cross_entopry = tf.reduce_mean(cross_entopry)

            abs_loss = tf.abs(cell - self.outputs)
            abs_loss = tf.reduce_mean(abs_loss)

            s = self.scale_factor
            loss = s*cross_entopry + (1.0-s)*abs_loss

        # Build vars_to_save.
        trainable_vars = tf.trainable_variables()
        moving_avg_vars = tf.moving_average_variables()
        self.vars_to_save = trainable_vars + moving_avg_vars

        with tf.variable_scope("train"):
            optimizer = tf.train.AdamOptimizer(
                            learning_rate=self.learning_rate,
                            beta1=0.5,
                        )

            self.train_op = optimizer.minimize(loss, var_list=trainable_vars)

        # Build summaries.
        with tf.name_scope("tensorboard"):
            tf.summary.scalar("cross_entopry", cross_entopry)
            tf.summary.scalar("abs_loss", abs_loss)

        with tf.name_scope("histogram_summary"):
            for v in self.vars_to_save:
                tf.summary.histogram(v.name, v)

        self.merged_summary = tf.summary.merge_all()


    def train(self, checkpoint=None, start_step=0):
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
            sess.run(self.valid_iterator.initializer)

            if checkpoint:
                print("Restoring:", checkpoint)
                saver.restore(sess, checkpoint)

            for i in itertools.count(start=start_step):
                feed_dict = {
                    self.training: True,
                }

                sess.run([self.train_op], feed_dict=feed_dict)

                if i % self.save_every == 0:
                    # Get validation data.
                    valid_data = sess.run(self.next_valid_data)

                    ops = [self.merged_summary, self.cell, self.outputs]
                    feed_dict = {self.x: valid_data}

                    run_options = tf.RunOptions(
                                      trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()


                    summary_str, cell, cell_infer = sess.run(
                       ops,
                       feed_dict=feed_dict,
                       options=run_options,
                       run_metadata=run_metadata,
                    )

                    file_writer.add_run_metadata(
                        run_metadata, "step_{}".format(i)
                    )

                    file_writer.add_summary(summary_str, i)
                    saver.save(sess, saver_name, global_step=i)

                    with open("{}/cell.txt".format(sample_dir), "w") as f:
                        cmin, cmax = self.dataset.cell_length_scale
                        cell = [(cmax-cmin)*x + cmin for x in cell]
                        cell_infer = [(cmax-cmin)*x + cmin for x in cell_infer]

                        for c, ci in zip(cell, cell_infer):
                            f.write("{:.3f} {:.3f}, {:.3f} {:.3f}, {:.3f} {:.3f}\n".format(
                                    c[0], ci[0],
                                    c[1], ci[1],
                                    c[2], ci[2],
                                )
                            )

                print("{}/{}  ITER: {}".format(logdir, date, i))


    def inference(self, griddata_folder, ckpt):
        sample_dir = "{}/samples-{}".format(self.logdir, self.date)

        # Remove useless folder.
        try:
            shutil.rmtree(sample_dir)
        except:
            print("error on os.mkdir?")

        files = glob.glob("{}/*.griddata".format(griddata_folder))

        saver = tf.train.Saver(var_list=self.vars_to_save, max_to_keep=1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            print("Restoring checkpoint: {}".format(ckpt))
            saver.restore(sess, ckpt)

            print("Inference step starts.")
            for i, gridfile in enumerate(files):
                # =====================================================
                # WARNING: Manually normalize. should be checked later.
                # Checked at 2018-02-20.
                # =====================================================
                grid = np.fromfile(gridfile, dtype=np.float32)
                grid = grid.reshape([1] + self.dataset.shape)
                # Normalize.
                mine, maxe = self.dataset.energy_scale

                grid[grid > maxe] = maxe
                grid[grid < mine] = mine

                grid = (grid-mine) / (maxe-mine)
                if self.dataset.invert:
                    grid = 1.0 - grid

                grid = grid.astype(np.float32)

                feed_dict = {
                    self.x[1]: grid,
                }

                outputs = sess.run(self.outputs, feed_dict=feed_dict)

                cmin, cmax = self.dataset.cell_length_scale
                output = (cmax-cmin)*outputs[0, ...] + cmin

                outfile = ".".join(gridfile.split(".")[:-1]) + ".grid"

                with open(outfile, "w") as f:
                    f.write("CELL_PARAMETERS {:.3f} {:.3f} {:.3f}\n".format(
                        output[0], output[1], output[2]))

                    f.write("CELL_ANGLES 90 90 90\n")
                    f.write("GRID_NUMBERS {s} {s} {s}\n".format(
                        s=self.dataset.shape[0])
                    )

                print(output)
                print("ITER: {}".format(i))
