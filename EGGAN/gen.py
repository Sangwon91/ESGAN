import argparse
import functools
import glob
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf

from model import EGGAN
from config import (ArgumentParser,
                    make_eggan_arg_parser,
                    write_config_log,
                    make_args_from_config,
                    find_config_from_checkpoint)
from dataset import EnergyGridTupleDataset

def main():
    # Custom argparser.
    gen_parser = ArgumentParser()
    gen_parser.add_argument("--checkpoint", type=str)
    gen_parser.add_argument("--n_samples", type=int)
    gen_parser.add_argument("--savedir", type=str)
    gen_parser.add_argument("--device", type=str)
    gen_parser.add_argument("--interpolate", action="store_true")

    # Parse args for gen.py
    gen_args = gen_parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gen_args.device

    parser = make_eggan_arg_parser()

    config = find_config_from_checkpoint(gen_args.checkpoint)
    # Parse original configs
    args = make_args_from_config(config)
    args = parser.parse_args(args)

    dataset = EnergyGridTupleDataset(
        path=args.dataset_path,
        rotate=args.rotate,
        shape=args.voxel_size,
        move=args.move,
        prefetch_size=256,
        shuffle_size=10000,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        cell_length_scale=args.cell_length_scale,
        invert=args.invert,
    )

    eggan = EGGAN(
        dataset=dataset,
        logdir=args.logdir,
        save_every=args.save_every,
        batch_size=50,
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
        in_temper=args.in_temper,
        feature_matching=args.feature_matching,
    )

    if gen_args.interpolate:
        eggan.interpolate_samples(
            gen_args.savedir,
            gen_args.checkpoint,
            gen_args.n_samples,
        )
    else:
        eggan.generate_samples(
            gen_args.savedir,
            gen_args.checkpoint,
            gen_args.n_samples
        )


if __name__ == "__main__":
    main()
