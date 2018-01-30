import os
import functools

import numpy as np
import tensorflow as tf

from model import WGANGP
from config import make_arg_parser, write_config_log
from dataset import make_energy_grid_dataset

from utils import write_visit_input

def main():
    parser = make_arg_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    energy_scale = args.energy_scale

    output_writer = functools.partial(
        write_visit_input,
        energy_scale=energy_scale,
        invert=args.invert)

    dataset = make_energy_grid_dataset(
        path=args.dataset_path,
        extension=args.extension,
        move=args.move,
        rotate=args.rotate,
        shape=args.voxel_size,
        prefetch_size=300,
        shuffle_size=300,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        invert=args.invert,
    )

    wgan_gp = WGANGP(
        dataset=dataset,
        logdir=args.logdir,
        output_writer=output_writer,
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
        n_critics=args.n_critics,
        gp_lambda=args.gp_lambda,
    )

    write_config_log(args, wgan_gp.date)

    wgan_gp.train()


if __name__ == "__main__":
    main()
