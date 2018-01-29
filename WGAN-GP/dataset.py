import glob
import random

import numpy as np
import tensorflow as tf

def make_energy_grid_dataset(*,
        path,
        shape,
        invert=True,
        rotate=True,
        move=True,
        extension="griddata",
        energy_limit=[-5000.0, 5000.0],
        energy_scale=[-6000.0, 6000.0],
        shuffle_size=None,
        prefetch_size=None,
        shared_name=None):

    if type(shape) not in (list, tuple):
        shape = [shape, shape, shape, 1]

    def _parse_grid(x):
        # Load data from the file.
        x = tf.read_file(x)
        x = tf.decode_raw(x, out_type=tf.float32)

        # reshape to 3D with 1 channel (total 4D).
        x = tf.reshape(x, shape)

        # Drop NaN in array.
        # if x[...] is NaN, it becames max_energy.
        x = tf.where(tf.is_nan(x), energy_limit[1]*tf.ones_like(x), x)

        # Normalize.
        # Infite energy to zero.
        x = tf.clip_by_value(x, energy_limit[0], energy_limit[1])
        x = (x - energy_scale[0]) / (energy_scale[1] - energy_scale[0])

        if invert:
            x = 1.0 - x

        if move:
            # Data augmentation
            n = tf.random_uniform(
                [1], minval=0, maxval=shape[0], dtype=tf.int32)[0]
            x = tf.concat([x[n:, :, :, :], x[:n, :, :, :]], axis=0)

            n = tf.random_uniform(
                [1], minval=0, maxval=shape[0], dtype=tf.int32)[0]
            x = tf.concat([x[:, n:, :, :], x[:, :n, :, :]], axis=1)

            n = tf.random_uniform(
                [1], minval=0, maxval=shape[0], dtype=tf.int32)[0]
            x = tf.concat([x[:, :, n:, :], x[:, :, :n, :]], axis=2)

        if rotate:
            pos = tf.constant([
                [0, 1, 2, 3],
                [2, 0, 1, 3],
                [1, 2, 0, 3]
            ], dtype=tf.int32)

            n = tf.random_uniform(
                [1], minval=0, maxval=3, dtype=tf.int32)[0]

            pos = pos[n]

            # Rotation.
            x = tf.transpose(x, pos)

        x.set_shape(shape)
        return x

    with tf.variable_scope("dataset"):
        # Build dataset.
        voxel_path_expression = "{}/*.{}".format(path, extension)

        dataset = tf.data.Dataset.list_files(voxel_path_expression)
        dataset = dataset.repeat()

        if shuffle_size:
            dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.map(_parse_grid, num_parallel_calls=2)

        if prefetch_size:
            dataset = dataset.prefetch(prefetch_size)

    return dataset


def make_egrid_tuple_dataset(
        path,
        shape,
        invert=True,
        move=True,
        rotate=False,
        energy_limit=[-3000.0, 5000.0],
        energy_scale=[-3000.0, 5000.0],
        cell_length_scale=[3.0, 50.0],
        shuffle_size=None,
        prefetch_size=None,
        shared_name=None):

    shape = [shape, shape, shape, 1]

    grid_list = glob.glob("{}/*.grid".format(path))
    griddata_list = [x + "data" for x in grid_list]

    grid_set = tf.data.Dataset.from_tensor_slices(grid_list).repeat()
    data_set = tf.data.Dataset.from_tensor_slices(griddata_list).repeat()

    def _parse_cell(path):
        data = dict()
        with open(path, "r") as f:
            for line in f:
                tokens = line.split()

                key = tokens[0]
                val = [float(x) for x in tokens[1:]]

                data[key] = val

        cmin = cell_length_scale[0]
        cmax = cell_length_scale[1]
        cell_lengths = [
            (x - cmin) / (cmax - cmin) for x in data["CELL_PARAMETERS"]]

        #amin = cell_angle_scale[0]
        #amax = cell_angle_scale[1]
        #cell_angles = [
        #    (x - amin) / (amax - amin) for x in data["CELL_ANGLES"]]

        #return np.array(cell_lengths + cell_angles, dtype=np.float32)
        return np.array(cell_lengths, dtype=np.float32)

    def _parse_grid(x):
        # Load data from the file.
        x = tf.read_file(x)
        x = tf.decode_raw(x, out_type=tf.float32)

        # reshape to 3D with 1 channel (total 4D).
        x = tf.reshape(x, shape)

        # Drop NaN in array.
        # if x[...] is NaN, it becames max_energy.
        x = tf.where(tf.is_nan(x), energy_limit[1]*tf.ones_like(x), x)

        # Normalize.
        # Infite energy to zero.
        x = tf.clip_by_value(x, energy_limit[0], energy_limit[1])
        x = (x - energy_scale[0]) / (energy_scale[1] - energy_scale[0])

        if invert:
            x = 1.0 - x

        if move:
            # Data augmentation
            n = tf.random_uniform(
                [1], minval=0, maxval=shape[0], dtype=tf.int32)[0]
            x = tf.concat([x[n:, :, :, :], x[:n, :, :, :]], axis=0)

            n = tf.random_uniform(
                [1], minval=0, maxval=shape[0], dtype=tf.int32)[0]
            x = tf.concat([x[:, n:, :, :], x[:, :n, :, :]], axis=1)

            n = tf.random_uniform(
                [1], minval=0, maxval=shape[0], dtype=tf.int32)[0]
            x = tf.concat([x[:, :, n:, :], x[:, :, :n, :]], axis=2)

        if rotate:
            pos = tf.constant([
                [0, 1, 2, 3],
                [2, 0, 1, 3],
                [1, 2, 0, 3]
            ], dtype=tf.int32)

            n = tf.random_uniform(
                [1], minval=0, maxval=3, dtype=tf.int32)[0]

            pos = pos[n]

            # Rotation.
            x = tf.transpose(x, pos)

        x.set_shape(shape)
        return x

    grid_set = grid_set.map(
        lambda path: tf.py_func(_parse_cell, [path], tf.float32)
    )

    data_set = data_set.map(_parse_grid)

    dataset = tf.data.Dataset.zip((grid_set, data_set))

    if shuffle_size:
        dataset = dataset.shuffle(shuffle_size)

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset

if __name__ == "__main__":
    pass
