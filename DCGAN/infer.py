import os
import sys
import glob
import shutil
import functools
import argparse

import numpy as np
import tensorflow as tf

from model import Frac2Cell
from config import (ArgumentParser,
                    make_frac2cell_arg_parser,
                    write_config_log,
                    make_args_from_config)
from dataset import make_egrid_tuple_dataset

from utils import write_griday_input

def prepare_sample_generation(config, griddata_folder):
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
    #ann_folder = "{}/ann-{}".format(config_folder, date)
    expression = "{}/save-{}-*".format(config_folder, date)
    ckpts = glob.glob(expression)

    try:
        print(">>>>>> Copying ckeckpoints...")
        for f in ckpts:
            print("Copying", f, "...")
            shutil.copy2(f, griddata_folder)
    except Exception as e:
        raise Exception(str(e) + ", Terminate program")

    ckpt = ".".join(ckpts[0].split(".")[:-1])
    ckpt = griddata_folder + "/" + ckpt.split("/")[-1]

    return ckpt


def main():
    gen_parser = ArgumentParser()
    gen_parser.add_argument("--config", type=str)
    gen_parser.add_argument("--griddata_folder", type=str)
    gen_parser.add_argument("--device", type=str)

    gen_args = gen_parser.parse_args()

    parser = make_frac2cell_arg_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = gen_args.device

    # Parse configs
    args = make_args_from_config(gen_args.config)
    args = parser.parse_args(args)

    energy_scale = args.energy_scale

    output_writer = functools.partial(
        write_griday_input,
        invert=args.invert,
        energy_scale=energy_scale,
        cell_length_scale=args.cell_length_scale,
    )

    dataset = make_egrid_tuple_dataset(
        path=args.dataset_path,
        rotate=args.rotate,
        shape=args.voxel_size,
        move=args.move,
        prefetch_size=256,
        shuffle_size=256,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        cell_length_scale=args.cell_length_scale,
        invert=args.invert,
    )

    validset = make_egrid_tuple_dataset(
        path=args.validset_path,
        rotate=args.rotate,
        move=args.move,
        shape=args.voxel_size,
        prefetch_size=256,
        shuffle_size=256,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        cell_length_scale=args.cell_length_scale,
        invert=args.invert,
    )

    frac2cell = Frac2Cell(
        dataset=dataset,
        validset=validset,
        logdir=args.logdir,
        batch_size=args.batch_size,
        voxel_size=args.voxel_size,
        rate=args.rate,
        top_size=args.top_size,
        filter_unit=args.filter_unit,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        scale_factor=args.scale_factor,
    )

    ckpt = prepare_sample_generation(gen_args.config, gen_args.griddata_folder)

    frac2cell.inference(gen_args.griddata_folder, ckpt)


if __name__ == "__main__":
    main()
