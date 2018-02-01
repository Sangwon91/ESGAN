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


        x.set_shape(shape)

        return x

    grid_set = grid_set.map(
        lambda path: tf.py_func(_parse_cell, [path], tf.float32)
    )

    data_set = data_set.map(_parse_grid)

    def _rotate_helper(grid, data):
        nn = tf.random_uniform(
             [1], minval=0, maxval=3, dtype=tf.int32)[0]

        pos1 = tf.constant([
                  [0, 1, 2, 3], # z-->z, y-->y, x-->x
                  [2, 0, 1, 3], # z-->x, y-->z, x-->y
                  [1, 2, 0, 3]  # z-->y, y-->x, x-->z
              ], dtype=tf.int32)

        pos1 = pos1[nn]

        pos2 = tf.constant([
                  [0, 1, 2], # x-->x, y-->y, z-->z
                  [1, 2, 0], # x-->y, y-->z, z-->x
                  [2, 0, 1]  # x-->z, y-->x, z-->y
              ], dtype=tf.int32)

        pos2 = pos2[nn]

        i = pos2[0]
        j = pos2[1]
        k = pos2[2]

        # Because the energy grid contains data as z, y, x order
        # (e.g., energy at x, y, z is data[z, y, x])
        #
        # So if the value of pos is [1, 2, 0] then the transposed energy grid
        # is data[y, x, z].
        # It means z axis becomes y, y axis becomes x, and the x axis becomes z
        # start --> end
        #     z --> y
        #     y --> x
        #     x --> z
        #
        # On the other hand, the cell lengths are stored as x, y, z order.
        # So if you transpose cell with the pos = [1, 2, 0], you get
        # cell[y, z, x].
        # start --> end
        #     x --> y
        #     y --> z
        #     z --> x
        # So you should use defferent pos, pos2

        grid = tf.stack([
                    grid[i],
                    grid[j],
                    grid[k]
               ])

        data = tf.transpose(data, pos1)

        return (grid, data)

    dataset = tf.data.Dataset.zip((grid_set, data_set))

    if rotate:
        dataset = dataset.map(_rotate_helper)

    if shuffle_size:
        dataset = dataset.shuffle(shuffle_size)

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    return dataset


if __name__ == "__main__":
    from utils import write_griday_input

    dataset = make_egrid_tuple_dataset(
        path=".",
        shape=32,
        invert=True,
        move=True,
        rotate=True,
        energy_limit=[-4000.0, 5000.0],
        energy_scale=[-4000.0, 5000.0],
        cell_length_scale=[0.0, 50.0],
        shuffle_size=None,
        prefetch_size=None,
        shared_name=None,
    )

    iterator = dataset.batch(32).make_initializable_iterator()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(iterator.initializer)
        cells, grids = sess.run(iterator.get_next())

        for i, (cell, grid) in enumerate(zip(cells, grids)):
            stem = "test_{:02d}".format(i)
            write_griday_input(
                stem=stem,
                grid_tuple=(cell, grid),
                size=32,
                invert=True,
                energy_scale=[-4000.0, 5000.0],
                cell_length_scale=[0.0, 50.0],
                save_dir="test"
            )
