import os
import argparse

import numpy as np
import tensorflow as tf

from utils import write_visit_input

from model import DCGAN
from dataset import make_energy_grid_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str)
    parser.add_argument("--logdir", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    batch_size = 16

    dataset = make_energy_grid_dataset(
        path="/home/FRAC32/IZA",
        extension="griddata",
        rotate=False,
        shape=32,
        prefetch_size=300,
        batch_size=batch_size,
        shuffle_size=300,
        energy_limit=[-5000.0, 5000.0],
        energy_scale=[-6000.0, 6000.0],
        invert=True,
    )

    dcgan = DCGAN(
        dataset=dataset,
        logdir=args.logdir,
        batch_size=batch_size,
        z_size=500,
        voxel_size=32,
        bottom_size=4,
        bottom_filters=256,
        #bottom_filters=128,
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

    dcgan.train()


if __name__ == "__main__":
    main()
