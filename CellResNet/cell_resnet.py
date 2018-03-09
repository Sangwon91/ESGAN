import os
import tempfile
import functools

import numpy as np
import tensorflow as tf

from ..EGGAN.model import CellResNet
from ..EGGAN.config import (make_cell_resnet_arg_parser,
                            write_config_log,
                            cache_ckpt_from_config)
from ..EGGAN.dataset import EnergyGridTupleDataset

def main():
    parser = make_cell_resnet_arg_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    energy_scale = args.energy_scale

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

    validset = EnergyGridTupleDataset(
        path=args.validset_path,
        rotate=args.rotate,
        move=args.move,
        shape=args.voxel_size,
        prefetch_size=256,
        shuffle_size=10000,
        energy_limit=args.energy_limit,
        energy_scale=args.energy_scale,
        cell_length_scale=args.cell_length_scale,
        invert=args.invert,
    )

    cell_resnet = CellResNet(
        dataset=dataset,
        validset=validset,
        logdir=args.logdir,
        batch_size=args.batch_size,
        filter_unit=args.filter_unit,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        scale_factor=args.scale_factor,
    )

    write_config_log(args, cell_resnet.date)

    ckpt = None
    step = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.restore_config:
            ckpt, step = cache_ckpt_from_config(
                             config=args.restore_config,
                             cache_folder=temp_dir,
                         )

        cell_resnet.train(checkpoint=ckpt, start_step=step)


if __name__ == "__main__":
    main()
