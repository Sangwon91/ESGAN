import os
import functools

import numpy as np
import tensorflow as tf

from model import Frac2Cell
from config import make_frac2cell_arg_parser, write_config_log
from dataset import make_egrid_tuple_dataset

from utils import write_griday_input

def main():
    parser = make_frac2cell_arg_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

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

    write_config_log(args, frac2cell.date)

    frac2cell.train()


if __name__ == "__main__":
    main()
