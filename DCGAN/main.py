import os
import tempfile
import functools

import numpy as np
import tensorflow as tf

from model import DCGAN
from config import (make_arg_parser,
                    write_config_log,
                    cache_ckpt_from_config)
from dataset import EnergyGridDataset

def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

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
        shuffle_size=10000,
    )

    dcgan = DCGAN(
        dataset=dataset,
        logdir=args.logdir,
        save_every=args.save_every,
        batch_size=args.batch_size,
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

    write_config_log(args, dcgan.date)

    ckpt = None
    step = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.restore_config:
            ckpt, step = cache_ckpt_from_config(
                             config=args.restore_config,
                             cache_folder=temp_dir,
                         )

        dcgan.train(checkpoint=ckpt, start_step=step)


if __name__ == "__main__":
    main()
