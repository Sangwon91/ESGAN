import os
import argparse
import functools

import numpy as np
import tensorflow as tf

from model import DCGAN
from dataset import make_energy_grid_dataset

from utils import write_visit_input

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--n_samples", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    energy_scale = [-3000.0, 5000.0]

    output_writer = functools.partial(
        write_visit_input, energy_scale=energy_scale)

    dataset = make_energy_grid_dataset(
        path="/home/FRAC32/PCOD_HIGH",
        extension="griddata",
        rotate=False,
        shape=32,
        prefetch_size=300,
        shuffle_size=300,
        energy_limit=[-3000.0, 5000.0],
        energy_scale=energy_scale,
        invert=True,
    )

    dcgan = DCGAN(
        dataset=dataset,
        logdir=args.sample_dir,
        output_writer=output_writer,
        save_every=100,
        batch_size=16,
        z_size=1024,
        voxel_size=32,
        bottom_size=4,
        bottom_filters=256,
        rate=0.5,
        top_size=4,
        filter_unit=32,
        g_learning_rate=0.0001,
        d_learning_rate=0.00001,
        minibatch=False,
        minibatch_kernel_size=500,
        minibatch_dim_per_kernel=5,
        l2_loss=False,
        train_gen_per_disc=1,
    )

    dcgan.generate_samples(args.sample_dir, args.ckpt, int(args.n_samples))


if __name__ == "__main__":
    main()