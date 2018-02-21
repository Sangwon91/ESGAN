import os
import glob
import shutil
import argparse
import functools

import numpy as np
import tensorflow as tf

from model import DCGAN
from config import (ArgumentParser,
                    make_arg_parser,
                    write_config_log,
                    make_args_from_config)
from dataset import EnergyGridDataset

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
    ann_folder = "{}/ann-{}".format(config_folder, date)
    expression = "{}/save-{}-*".format(config_folder, date)
    ckpts = glob.glob(expression)

    try:
        print(">>>>>> Removing folder if exist...")
        shutil.rmtree(ann_folder)
    except Exception as e:
        print(e, "but keep going")

    try:
        print(">>>>>> Making folder...")
        os.makedirs(ann_folder)
    except Exception as e:
        print(e, "but keep going")

    try:
        print(">>>>>> Copying ckeckpoint...")
        for f in ckpts:
            shutil.copy2(f, ann_folder)
    except Exception as e:
        raise Exception(str(e) + ", Terminate program")

    ckpt = ".".join(ckpts[0].split(".")[:-1])
    ckpt = ckpt.split("/")[-1]
    ckpt = ann_folder + "/" + ckpt

    return ann_folder, ckpt

def main():
    # Custom argparser.
    gen_parser = ArgumentParser()
    gen_parser.add_argument("--config", type=str)
    gen_parser.add_argument("--n_samples", type=str)
    gen_parser.add_argument("--device", type=str)

    # Parse args for gen.py
    gen_args = gen_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gen_args.device

    parser = make_arg_parser()

    # Parse original configs
    args = make_args_from_config(gen_args.config)
    args = parser.parse_args(args)

    energy_scale = args.energy_scale

    dataset = EnergyGridDataset(
        path=args.dataset_path,
        shape=args.voxel_size,
        invert=args.invert,
        rotate=args.rotate,
        move=args.move,
        extension=args.extension,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        prefetch_size=300,
        shuffle_size=300,
    )

    dcgan = DCGAN(
        dataset=dataset,
        logdir=args.logdir,
        save_every=args.save_every,
        batch_size=30,
        z_size=args.z_size,
        voxel_size=args.voxel_size,
        bottom_size=args.bottom_size,
        bottom_filters=args.bottom_filters,
        rate=args.rate,
        top_size=args.top_size,
        filter_unit=args.filter_unit,
        g_learning_rate=args.g_learning_rate,
        d_learning_rate=args.d_learning_rate,
        minibatch=args.minibatch,
        minibatch_kernel_size=args.minibatch_kernel_size,
        minibatch_dim_per_kernel=args.minibatch_dim_per_kernel,
        l2_loss=args.l2_loss,
        train_gen_per_disc=args.train_gen_per_disc,
    )

    ann_folder, ckpt = prepare_sample_generation(gen_args.config)

    dcgan.interpolate_samples(ann_folder, ckpt, int(gen_args.n_samples))


if __name__ == "__main__":
    main()
