import os
import argparse

from model2 import BEGAN
from dataset import make_energy_grid_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str)
    parser.add_argument("--logdir", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # BEGAN test. =============================================================
    config = BEGAN.ConfigProto()

    config.h = 8*8*8*32
    config.n = 32
    config.size = 64
    config.repeats = 2
    config.channels = 1
    config.gamma = 1.0
    config.lam = 0.0001
    config.learning_rate = 0.00001
    config.logdir = args.logdir
    config.save_every = 20
    config.train_steps = 1000000000
    config.vanishing_steps = 0
    config.batch_size = 8

    from dataset import make_energy_grid_dataset

    config.dataset = make_energy_grid_dataset(
        #path="/tmp/IZA",
        #path="/home/lsw/IZA_VOXEL",
        #path="/home/lsw/IZA_KH",
        #path="/home/lsw/RWY",
        path="/home/lsw/BOF",
        shape=64,
        prefetch_size=100,
        batch_size=8,
        shuffle_size=10000,
        energy_range=[-3000, 5000],
    )

    began = BEGAN(config)

    began.train()


if __name__ == "__main__":
    main()
