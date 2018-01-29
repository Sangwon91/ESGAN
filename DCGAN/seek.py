import os
import glob
import shutil
import argparse
import functools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import Generator, Discriminator
from config import (ArgumentParser,
                    make_arg_parser,
                    write_config_log,
                    make_args_from_config)
from dataset import make_energy_grid_dataset

def prepare_sample_generation(config):
    """
    1. Generate sample dir.
    2. Copy lastest checkpoint.

    """
    # Get file name (not a path)
    config_name = config.split("/")[-1]
    # Get parent path
    config_folder = "/".join(config.split("/")[:-1])
    # Extract path
    date = "-".join(config_name.split("-")[1:])

    # Make sample dir
    seek_folder = "{}/seek-{}".format(config_folder, date)
    expression = "{}/save-{}-*".format(config_folder, date)
    ckpts = glob.glob(expression)

    try:
        print(">>>>>> Removing folder if exist...")
        shutil.rmtree(seek_folder)
    except Exception as e:
        print(e, "but keep going")

    try:
        print(">>>>>> Making folder...")
        os.makedirs(seek_folder)
    except Exception as e:
        print(e, "but keep going")

    try:
        print(">>>>>> Copying ckeckpoint...")
        for f in ckpts:
            shutil.copy2(f, seek_folder)
    except Exception as e:
        raise Exception(str(e) + ", Terminate program")

    ckpt = ".".join(ckpts[0].split(".")[:-1])
    ckpt = ckpt.split("/")[-1]
    ckpt = seek_folder + "/" + ckpt

    return seek_folder, ckpt

def main():
    # Custom argparser.
    gen_parser = ArgumentParser()
    gen_parser.add_argument("--config", type=str)
    gen_parser.add_argument("--device", type=str)

    # Parse args for gen.py
    gen_args = gen_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gen_args.device

    parser = make_arg_parser()

    # Parse original configs
    args = make_args_from_config(gen_args.config)
    args = parser.parse_args(args)

    seek_folder, ckpt = prepare_sample_generation(gen_args.config)

    z = tf.Variable(
            np.random.normal(-1.0, 1.0, size=[1, args.z_size]),
            dtype=tf.float32,
            name="noise",
        )

    z_clip = tf.clip_by_value(z, clip_value_min=-1.0, clip_value_max=1.0)

    generator = Generator(
        batch_size=1,
        z_size=args.z_size,
        voxel_size=args.voxel_size,
        bottom_size=args.bottom_size,
        bottom_filters=args.bottom_filters,
        z=z_clip,
    )

    disc = Discriminator(
        x=generator.outputs,
        batch_size=args.batch_size,
        voxel_size=args.voxel_size,
        rate=0.0,
        top_size=args.top_size,
        filter_unit=args.filter_unit,
        minibatch=False,
        minibatch_kernel_size=None,
        minibatch_dim_per_kernel=None,
        reuse=False,
        name="discriminator",
    )

    rwy = np.fromfile("/home/FRAC32/RWY/RWY.griddata", dtype=np.float32)
    rwy = rwy.reshape([1, 32, 32, 32, 1])
    rwy[rwy > 5000.0] = 5000.0
    rwy[rwy < -3000.0] = -3000.0
    tar = tf.constant(rwy, dtype=tf.float32)

    lower, upper = args.energy_scale
    energy = (upper-lower) * (1.0-generator.outputs) + lower
    diff = tf.reduce_mean(tf.abs(tar - energy))
    """
    cut_energy = tf.where(
                     energy < -2500.0,
                     upper*tf.ones_like(energy),
                     energy
                 )
    """
    prob = tf.reduce_mean(disc.outputs)
    henry = tf.reduce_mean(tf.exp(-energy / 300.0))

    optimizer = tf.train.AdamOptimizer(learning_rate=10.0)
    #henry = tf.reduce_mean(tf.exp(-cut_energy / 300.0))
    #train_op = optimizer.minimize(-henry, var_list=[z])

    void_fraction = tf.reduce_sum(tf.nn.sigmoid(-(energy - 4900.0))) / 32.0**3
    train_op = optimizer.minimize(diff, var_list=[z])

    vars_to_save = tf.trainable_variables() + tf.moving_average_variables()
    vars_to_save = [v for v in vars_to_save if "generator" in v.name]
    saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=1)

    idx = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)

        for i in range(1000000):
            grid, kh, vf, p, zs, d, _ = sess.run(
                [energy, henry, void_fraction, prob, z_clip, diff, train_op],
            )

            if i % 200 == 0:
                save_name = "{}/kh_max_{}.grid".format(seek_folder, idx)

                with open(save_name, "w") as f:
                    f.write("CELL_PARAMETERS 1 1 1\n")
                    f.write("CELL_ANGLES 90 90 90\n")
                    f.write("GRID_NUMBERS 32 32 32")

                grid.tofile(save_name + "data")

                print("ITER:", i, "KH:", kh, "VF:", vf, "P:", p, "D:", d)

                idx += 1

    zs.tofile("z.dat")

if __name__ == "__main__":
    main()
