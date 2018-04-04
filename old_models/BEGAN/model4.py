import os
import math
import datetime

import numpy as np
import tensorflow as tf

from ops import *
from utils import write_visit_input

class Freezer:
    __frozen = False

    def __setattr__(self, key, value):
        if self.__frozen and not hasattr(self, key):
            raise AttributeError("No attr error. key={}".format(key))
        else:
            object.__setattr__(self, key, value)

    def _freeze(self):
        self.__frozen = True

    def _unfreeze(self):
        self.__frozen = False


class Encoder:
    def __init__(self, config):
        self.config = config

        name = self.config.name
        reuse = self.config.reuse
        with tf.variable_scope(name, reuse=reuse):
            self._build()


    def _build(self):
        # Build NN.
        n_loops = int(math.log2(self.config.size / 8)) + 1

        conv_names = ("conv_3d_{}".format(i) for i in range(1000))
        vans_names = ("vanishing_{}".format(i) for i in range(1000))

        # Set alias.
        carry = self.config.carry
        repeats = self.config.repeats
        vanishing_residual = self.config.vanishing_residual
        n = self.config.n
        cur_n = n # The value of current n.

        # First layer.
        x = self.config.x
        x = conv3d(x, filters=n, name=next(conv_names))

        for l in range(n_loops):
            for r in range(repeats):
                next_x = conv3d(x, filters=cur_n, name=next(conv_names))

                # Apply vanishing residual.
                if is_same_shape(x, next_x):
                    next_x = vanishing_residual_layer(
                                 x, next_x, vanishing_residual,
                                 carry, next(vans_names),
                             )

                x = next_x

                if l != n_loops - 1:
                    if r == repeats - 2:
                        break

            # No need to down_sampling for the last layer.
            if l + 1 == n_loops:
                break

            # Expanding filters and down-sampling.
            cur_n += n

            x = conv3d(x, filters=cur_n, name=next(conv_names))
            x = down_sampling3d(x, scale=2)

        with tf.variable_scope("encoding"):
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, units=self.config.h, activation=None)

        self.inputs = self.config.x
        self.outputs = x


    class ConfigProto(Freezer):
        def __init__(self):
            self.x = None
            self.h = None
            self.n = None
            self.size = None
            self.repeats = 2
            self.vanishing_residual = None
            self.carry = None
            self.reuse = None
            self.name = "encoder"

            self._freeze()


class Decoder:
    def __init__(self, config):
        self.config = config

        name = self.config.name
        reuse = self.config.reuse

        with tf.variable_scope(name, reuse=reuse):
            self._build()


    def _build(self):
        n_loops = int(math.log2(self.config.size / 8))

        conv_names = ("conv_3d_{}".format(i) for i in range(1000))
        inj_names = ("injection_{}".format(i) for i in range(1000))
        vans_names = ("vanishing_{}".format(i) for i in range(1000))

        z = self.config.z
        n = self.config.n
        with tf.variable_scope("h0"):
            h0 = tf.nn.dropout(z, keep_prob=0.8)
            h0 = tf.layers.dense(h0, units=8*8*8*n, activation=None)
            h0 = tf.reshape(h0, shape=[-1, 8, 8, 8, n])

        # Set alias
        carry = self.config.carry
        repeats = self.config.repeats
        vanishing_residual = self.config.vanishing_residual

        scale = 1
        x = h0
        for l in range(n_loops):
            for _ in range(repeats):
                next_x = conv3d(x, filters=n, name=next(conv_names))

                if is_same_shape(x, next_x):
                    next_x = vanishing_residual_layer(
                                 x, next_x, vanishing_residual,
                                 carry, next(vans_names),
                             )

                x = next_x

            x = up_sampling3d(x, scale=2)

            # Skip connection.
            if l != n_loops - 1:
                with tf.variable_scope(next(inj_names)):
                    scale *= 2
                    scaled_h0 = up_sampling3d(h0, scale=scale)
                    x = tf.concat([x, scaled_h0], axis=4)


        # Final layer.
        channels = self.config.channels
        x = conv3d(x, filters=channels, activation=None, name=next(conv_names))

        self.logits = x

        x = tf.nn.sigmoid(x)

        self.inputs = self.config.z
        self.outputs = x


    class ConfigProto(Freezer):
        def __init__(self):
            self.z = None
            self.n = None
            self.size = None
            self.channels = 1
            self.repeats = 2
            self.vanishing_residual = None
            self.carry = None
            self.reuse = None
            self.name = "decoder"

            self._freeze()


class Generator:
    def __init__(self, config):
        self.config = config

        decoder = Decoder(config)

        self.inputs = decoder.inputs
        self.outputs = decoder.outputs


    class ConfigProto(Decoder.ConfigProto):
        def __init__(self):
            # Build super's args.
            super().__init__()
            # Change name "decoder" to "generator".
            self.name = "generator"


class Discriminator:
    def __init__(self, config):
        self.config = config

        name = self.config.name
        reuse = self.config.reuse

        e_conf = Encoder.ConfigProto()
        d_conf = Decoder.ConfigProto()

        for k, v in self.config.__dict__.items():
            if hasattr(e_conf, k):
                setattr(e_conf, k, v)
            if hasattr(d_conf, k):
                setattr(d_conf, k, v)

        e_conf.name = "encoder"
        d_conf.name = "decoder"

        with tf.variable_scope(name, reuse=reuse):
            encoder = Encoder(e_conf)

            d_conf.z = encoder.outputs
            decoder = Decoder(d_conf)

        self.inputs = encoder.inputs
        self.outputs = decoder.outputs

        # Additional info if exist.
        try:
            self.logits = decoder.logits
        except:
            pass


    class ConfigProto(Freezer):
        def __init__(self):
            self.x = None
            self.n = None
            self.size = None
            self.repeats = None
            self.channels = 1
            self.vanishing_residual = None
            self.carry = None
            self.h = None
            self.name = "discriminator"
            self.reuse = None

            self._freeze()


class BEGAN:
    def __init__(self, config):
        self.config = config

        with tf.variable_scope(self.config.name):
            self.carry = tf.placeholder_with_default(0.0,
                             shape=[], name="carry")

            self.vanishing_residual = tf.placeholder_with_default(False,
                                          shape=[], name="vanishing_residual")

            self._build()


    def _build(self):
        with tf.variable_scope("make_dataset"):
            self.iterator = self.config.dataset.make_initializable_iterator()
            self.next_batch = self.iterator.get_next()

        gen_conf = Generator.ConfigProto()
        disc_conf = Discriminator.ConfigProto()

        # Copy configurations.
        for k, v in self.config.__dict__.items():
            if hasattr(gen_conf, k):
                setattr(gen_conf, k, v)
            if hasattr(disc_conf, k):
                setattr(disc_conf, k, v)

        gen_conf.carry = self.carry
        gen_conf.vanishing_residual = self.vanishing_residual

        disc_conf.carry = self.carry
        disc_conf.vanishing_residual = self.vanishing_residual

        gen_conf.name = "generator"
        disc_conf.name = "discriminator"

        # Make a generator.
        h = self.config.h
        gen_conf.z = tf.placeholder(
            shape=[None, h],
            dtype=tf.float32,
            name="z",
        )
        self.generator = Generator(gen_conf)

        # save z to member var.
        self.z = gen_conf.z

        # Make a discriminator for real data.
        disc_conf.x = self.next_batch
        self.disc_real = Discriminator(disc_conf)

        # Make a Discriminator for fake data.
        disc_conf.x = self.generator.outputs
        disc_conf.reuse = True
        self.disc_fake = Discriminator(disc_conf)

        # Make losses.
        gamma = self.config.gamma
        lam = self.config.lam
        learning_rate = self.config.learning_rate

        real_in = self.disc_real.inputs
        real_out = self.disc_real.outputs

        fake_in = self.disc_fake.inputs
        fake_out = self.disc_fake.outputs

        k = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="k")

        with tf.variable_scope("loss/real"):
            real_diff = tf.abs(real_in - real_out)
            real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=real_in, logits=self.disc_real.logits)
            real_loss = tf.reduce_mean(real_loss)

        with tf.variable_scope("loss/fake"):
            fake_diff = tf.abs(fake_in - fake_out)
            fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=fake_in, logits=self.disc_fake.logits)
            fake_loss = tf.reduce_mean(fake_loss)

        with tf.variable_scope("loss/disc"):
            disc_loss = real_loss - k*fake_loss

        with tf.variable_scope("loss/gen"):
            gen_loss = tf.identity(fake_loss)


        with tf.variable_scope("loss/global"):
            global_loss = real_loss + tf.abs(gamma*real_loss - fake_loss)

        # Make training ops.
        train_vars = tf.trainable_variables()
        d_vars = [v for v in train_vars if "discriminator/" in v.name]
        g_vars = [v for v in train_vars if "generator/" in v.name]

        with tf.variable_scope("train/disc"):
            d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            d_train_op = d_opt.minimize(disc_loss, var_list=d_vars)

        with tf.variable_scope("train/gen"):
            g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            g_train_op = g_opt.minimize(gen_loss, var_list=g_vars)

        with tf.variable_scope("train/update_k"):
            #with tf.control_dependencies([d_train_op, g_train_op]):
            new_k = k + lam*(gamma*real_loss - fake_loss)
            new_k = tf.clip_by_value(new_k, 0.0, 1.0)

            update_k = tf.assign(k, new_k)

        self.train_ops = [d_train_op, g_train_op, update_k]
        #self.train_ops = [d_train_op]

        # Make summary ops.
        tf.summary.scalar("loss_summary/disc", disc_loss)
        tf.summary.scalar("loss_summary/gen", gen_loss)
        tf.summary.scalar("loss_summary/real", real_loss)
        tf.summary.scalar("loss_summary/fake", fake_loss)
        tf.summary.scalar("loss_summary/global", global_loss)

        tf.summary.scalar("misc/k", k)
        tf.summary.scalar("misc/abs/real", tf.reduce_mean(real_diff))
        tf.summary.scalar("misc/abs/fake", tf.reduce_mean(fake_diff))

        for var in train_vars:
            if "kernel" in var.name or "bias" in var.name:
                tf.summary.histogram(var.name, var)

        # Make shape to remove None in the shape.
        shape = real_diff.get_shape().as_list()
        shape[0] = self.config.batch_size

        real_diff.set_shape(shape)
        real_diffs = tf.unstack(real_diff)
        for i, diff in enumerate(real_diffs):
            tf.summary.histogram("real_diff_{}".format(i), diff)

        fake_diff.set_shape(shape)
        fake_diffs = tf.unstack(fake_diff)
        for i, diff in enumerate(fake_diffs):
            tf.summary.histogram("fake_diff_{}".format(i), diff)

        self.merged_summary = tf.summary.merge_all()

        # Make savers.
        self.date = datetime.datetime.now().isoformat()
        self.writer = tf.summary.FileWriter(
            "{}/summary_{}".format(self.config.logdir, self.date),
            tf.get_default_graph()
        )
        self.saver = tf.train.Saver(max_to_keep=5)

        self.sample_dir = "{}/samples_{}".format(self.config.logdir, self.date)
        self.ckpt_dir = "{}/ckpt_{}".format(self.config.logdir, self.date)

        # Make dir for samples
        try:
            os.makedirs(self.sample_dir)
        except Exception as e:
            print("MKDIR DIR ERROR:", e)

        # Log configuration to the logdir.
        config_name = "{}/config_{}".format(self.config.logdir, self.date)
        with open(config_name, "w") as f:
            for k, v in self.config.__dict__.items():
                f.write("{} = {}\n".format(k, v))


    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Initialize dataset
        self.sess.run(self.iterator.initializer)
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        h = self.config.h
        batch_size = self.config.batch_size
        train_steps = self.config.train_steps
        vanishing_steps = self.config.vanishing_steps

        carrys = iter(np.linspace(1.0, 0.0, vanishing_steps))

        for i in range(train_steps):
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, h])
            noise = noise.astype(np.float32)

            try:
                carry_val = next(carrys)
            except:
                carry_val = 0.0

            feed_dict = {
                self.z: noise,
                self.carry: carry_val,
                self.vanishing_residual: i < vanishing_steps,
            }

            self.sess.run(self.train_ops, feed_dict=feed_dict)

            if i % self.config.save_every == 0:
                run_options = tf.RunOptions(
                                  trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                ops = [
                    self.merged_summary,
                    self.generator.outputs,
                    self.disc_real.inputs,
                    self.disc_real.outputs,
                ]

                summary, samples, real_in, real_out = self.sess.run(
                    ops,
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata
                )

                self.writer.add_run_metadata(run_metadata, "step_{}".format(i))
                self.writer.add_summary(summary, i)

                path = "{}/model-{}.ckpt".format(self.ckpt_dir, i)

                self.saver.save(self.sess, path)

                # write samples.
                energy_range=[-3000, 5000]

                for j, sample in enumerate(samples):
                    stem = "sample_{}".format(j)
                    write_visit_input(
                        stem, sample, self.config.size,
                        energy_range, self.sample_dir)

                for j, sample in enumerate(real_in):
                    stem = "real_in_{}".format(j)
                    write_visit_input(
                        stem, sample, self.config.size,
                        energy_range, self.sample_dir)

                for j, sample in enumerate(real_out):
                    stem = "real_out_{}".format(j)
                    write_visit_input(
                        stem, sample, self.config.size,
                        energy_range, self.sample_dir)

            print("ITER:", i)


    class ConfigProto(Freezer):
        def __init__(self):
            self.batch_size = None
            self.h = None
            self.n = None
            self.size = None
            self.repeats = None
            self.channels = 1
            self.dataset = None
            self.name = "BEGAN"
            self.gamma = None
            self.lam = None # lambda
            self.learning_rate = None
            self.logdir = None
            self.save_every = None
            self.train_steps = 1000000000
            self.vanishing_steps = 10000

            self._freeze()


# UNIT TEST
def test():
    # Encoder test.
    config = Encoder.ConfigProto()

    config.x = tf.placeholder(
        shape=[None, 128, 128, 128, 3], dtype=tf.float32, name="x"
    )
    config.h = 64
    config.n = 32
    config.size = 128
    config.repeats = 2
    config.vanishing_residual = tf.placeholder(
        shape=[], dtype=tf.bool, name="vanishing_residual"
    )
    config.carry = tf.placeholder(shape=[], dtype=tf.float32, name="carry")
    config.reuse = None
    config.name = "ENCODER"

    encoder = Encoder(config)

    # Reuse test.
    config.x = tf.placeholder(
        shape=[None, 128, 128, 128, 3], dtype=tf.float32, name="x"
    )
    config.reuse = True
    encoder_1 = Encoder(config)

    tf.summary.FileWriter("test/model/encoder", tf.get_default_graph())

    # Decoder test.
    tf.reset_default_graph()

    config = Decoder.ConfigProto()

    config.z = tf.placeholder(shape=[None, 64], dtype=tf.float32, name="z")
    config.n = 32
    config.size = 128
    config.repeats = 2
    config.channels = 1
    config.vanishing_residual = tf.placeholder(
        shape=[], dtype=tf.bool, name="vanishing_residual"
    )
    config.carry = tf.placeholder(shape=[], dtype=tf.float32, name="carry")
    config.reuse = False
    config.name = "DECODER"

    decoder = Decoder(config)

    tf.summary.FileWriter("test/model/decoder", tf.get_default_graph())

    # Generator test.
    tf.reset_default_graph()

    config = Generator.ConfigProto()
    config.z = tf.placeholder(shape=[None, 64], dtype=tf.float32, name="z")
    config.n = 32
    config.size = 128
    config.repeats = 2
    config.channels = 1
    config.vanishing_residual = tf.placeholder(
        shape=[], dtype=tf.bool, name="vanishing_residual"
    )
    config.carry = tf.placeholder(shape=[], dtype=tf.float32, name="carry")
    config.reuse = False
    print(config.__dict__)

    generator = Generator(config)
    print(generator.outputs)

    tf.summary.FileWriter("test/model/generator", tf.get_default_graph())

    # Discriminator test.
    tf.reset_default_graph()

    config = Discriminator.ConfigProto()

    for k, v in config.__dict__.items():
        print(k, v)

    carry = tf.placeholder(shape=[], dtype=tf.float32, name="carry")
    vanishing_residual = tf.placeholder(
        shape=[], dtype=tf.bool, name="vanishing_residual")

    config.h = 64
    config.n = 32
    config.size = 64
    config.repeats = 2
    config.channels = 1
    config.vanishing_residual = vanishing_residual
    config.carry = carry

    config.x = tf.placeholder(
        shape=[None, 128, 128, 128, 3], dtype=tf.float32, name="x"
    )

    config.reuse = False
    config.name = "DISC"

    print(config.__dict__)

    disc = Discriminator(config)

    config.reuse = True
    disc = Discriminator(config)

    tf.summary.FileWriter("test/model/disc", tf.get_default_graph())

    # BEGAN test. =============================================================
    tf.reset_default_graph()

    config = BEGAN.ConfigProto()

    config.h = 512
    config.n = 32
    config.size = 64
    config.repeats = 2
    config.channels = 1
    config.gamma = 0.5
    config.lam = 0.001
    config.learning_rate = 0.000001
    config.logdir = "logdir"
    config.ckptdir = "ckptdir"
    config.sample_dir = "sampledir"
    config.save_every = 20
    config.train_steps = 1000000000
    config.vanishing_steps = 0
    config.batch_size = 8

    from dataset import make_energy_grid_dataset

    config.dataset = make_energy_grid_dataset(
        #path="/tmp/IZA",
        #path="/home/lsw/IZA_VOXEL",
        path="/home/lsw/IZA_KH",
        shape=64,
        prefetch_size=100,
        batch_size=8,
        shuffle_size=10000,
        energy_range=[-3000, 5000],
    )

    def vars_on_cpu(op):
        if "Variable" in op.type:
            return "/cpu:0"
        else:
            return "/gpu:0"

    #with tf.device(vars_on_cpu):
    began = BEGAN(config)

    try:
        began.train()
    except Exception as e:
        print(e)

    tf.summary.FileWriter("test/model/began", tf.get_default_graph())

    # Config test.
    try:
        config.any_key = "Hello"
    except Exception as e:
        print(e)
if __name__ == "__main__":
    test()
