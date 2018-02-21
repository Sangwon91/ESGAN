import glob
import random
import textwrap

import numpy as np
import tensorflow as tf

# Class version of dataset.
# Notice that this is not a child class of tf.data.Dataset.
class EnergyGridDataset:
    def __init__(self, *,
        path,
        shape,
        invert,
        rotate,
        move,
        extension,
        energy_limit,
        energy_scale,
        shuffle_size,
        prefetch_size,
        shared_name=None):

        # Build properties.
        self.path =  path
        if type(shape) not in (list, tuple):
            shape = [shape, shape, shape, 1]
        self.shape = shape
        self.invert = invert
        self.rotate = rotate
        self.move = move
        self.extension = extension
        self.energy_limit = energy_limit
        self.energy_scale = energy_scale
        self.shuffle_size = shuffle_size
        self.prefetch_size = prefetch_size
        self.shared_name = shared_name

        self._build_dataset()

    def _read_grid(self, x):
        # Load data from the file.
        x = tf.read_file(x)
        x = tf.decode_raw(x, out_type=tf.float32)

        # reshape to 3D with 1 channel (total 4D).
        x = tf.reshape(x, self.shape)

        return x

    def _normalize_grid(self, x):
        # Drop NaN in array.
        # if x[...] is NaN, it becames max_energy.
        lower_limit, upper_limit = self.energy_limit
        lower_scale, upper_scale = self.energy_scale

        x = tf.where(
                tf.is_nan(x),
                upper_limit*tf.ones_like(x), # True
                x                            # False
            )

        # Normalize.
        x = tf.clip_by_value(x, lower_limit, upper_limit)
        x = (x - lower_scale) / (upper_scale - lower_scale)

        return x

    def _invert_grid(self, x):
        return 1.0 - x

    def _move_grid(self, x):
        # Translate energy grid along axis.
        maxval = self.shape[0]

        n = tf.random_uniform(
            [1], minval=0, maxval=maxval, dtype=tf.int32)[0]
        x = tf.concat([x[n:, :, :, :], x[:n, :, :, :]], axis=0)

        n = tf.random_uniform(
            [1], minval=0, maxval=maxval, dtype=tf.int32)[0]
        x = tf.concat([x[:, n:, :, :], x[:, :n, :, :]], axis=1)

        n = tf.random_uniform(
            [1], minval=0, maxval=maxval, dtype=tf.int32)[0]
        x = tf.concat([x[:, :, n:, :], x[:, :, :n, :]], axis=2)

        x.set_shape(self.shape)

        return x

    def _rotate_grid(self, x):
        # Rotation.
        pos = tf.constant([
            [0, 1, 2, 3],
            [2, 0, 1, 3],
            [1, 2, 0, 3]
        ], dtype=tf.int32)

        n = tf.random_uniform(
            [1], minval=0, maxval=3, dtype=tf.int32)[0]

        pos = pos[n]
        x = tf.transpose(x, pos)

        x.set_shape(self.shape)

        return x

    def _parse_grid(self, x):
        x = self._read_grid(x)
        x = self._normalize_grid(x)

        if self.invert:
            x = self._invert_grid(x)

        if self.move:
            x = self._move_grid(x)

        if self.rotate:
            x = self._rotate_grid(x)

        return x

    def _build_dataset(self):
        # Build dataset.
        voxel_path_expression = "{}/*.{}".format(self.path, self.extension)

        dataset = tf.data.Dataset.list_files(voxel_path_expression)
        dataset = dataset.repeat()

        if self.shuffle_size:
            dataset = dataset.shuffle(self.shuffle_size)

        dataset = dataset.map(self._parse_grid, num_parallel_calls=1)

        if self.prefetch_size:
            dataset = dataset.prefetch(self.prefetch_size)

        self.dataset = dataset

    def write_visit_sample(self, *, x, stem, save_dir="."):
        lower, upper = self.energy_scale

        x = np.array(x.reshape([-1]))

        if self.invert:
            x = 1.0 - x

        x = (upper-lower)*x + lower

        # Make file name
        bov = save_dir + "/" + stem + ".bov"
        times = stem + ".times"

        size = self.shape[0]
        box = 1 # box=1 for the fractional coordinates.

        # Write header file.
        with open(bov, "w") as bovfile:
            bovfile.write(textwrap.dedent("""\
                TIME: 1.000000
                DATA_FILE: {}
                DATA_SIZE:     {size} {size} {size}
                DATA_FORMAT: FLOAT
                VARIABLE: data
                DATA_ENDIAN: LITTLE
                CENTERING: nodal
                BRICK_ORIGIN:        0  0  0
                BRICK_SIZE:       {box} {box} {box}""".format(
                times, size=size, box=box)
            ))
        # Write times file.
        x.tofile("{}/{}".format(save_dir, times))

    def write_sample(self, *, x, stem, save_dir="."):
        lower, upper = self.energy_scale

        x = np.array(x.reshape([-1]))

        if self.invert:
            x = 1.0 - x

        x = (upper-lower)*x + lower
        # Write times file.
        x.tofile("{}/{}.{}".format(save_dir, stem, self.extension))


class EnergyGridTupleDataset:
    def __init__(self, *,
        path,
        shape,
        invert,
        rotate,
        move,
        energy_limit,
        energy_scale,
        cell_length_scale,
        shuffle_size,
        prefetch_size,
        shared_name=None):

        # Build properties.
        self.path =  path
        if type(shape) not in (list, tuple):
            shape = [shape, shape, shape, 1]
        self.shape = shape
        self.invert = invert
        self.rotate = rotate
        self.move = move
        self.energy_limit = energy_limit
        self.energy_scale = energy_scale
        self.cell_length_scale = cell_length_scale
        self.shuffle_size = shuffle_size
        self.prefetch_size = prefetch_size
        self.shared_name = shared_name

        self._build_dataset()

    def _read_and_nomalize_cell(self, path):
        data = dict()
        with open(path, "r") as f:
            for line in f:
                tokens = line.split()

                key = tokens[0]
                val = [float(x) for x in tokens[1:]]

                data[key] = val

        cmin, cmax = self.cell_length_scale
        cell_lengths = [
            (x-cmin) / (cmax-cmin) for x in data["CELL_PARAMETERS"]]

        return np.array(cell_lengths, dtype=np.float32)

    def _read_grid(self, x):
        # Load data from the file.
        x = tf.read_file(x)
        x = tf.decode_raw(x, out_type=tf.float32)

        # reshape to 3D with 1 channel (total 4D).
        x = tf.reshape(x, self.shape)

        return x

    def _normalize_grid(self, x):
        # Drop NaN in array.
        # if x[...] is NaN, it becames max_energy.
        lower_limit, upper_limit = self.energy_limit
        lower_scale, upper_scale = self.energy_scale

        x = tf.where(
                tf.is_nan(x),
                upper_limit*tf.ones_like(x), # True
                x                            # False
            )

        # Normalize.
        x = tf.clip_by_value(x, lower_limit, upper_limit)
        x = (x - lower_scale) / (upper_scale - lower_scale)

        return x

    def _invert_grid(self, x):
        return 1.0 - x

    def _move_grid(self, x):
        # Translate energy grid along axis.
        maxval = self.shape[0]

        n = tf.random_uniform(
            [1], minval=0, maxval=maxval, dtype=tf.int32)[0]
        x = tf.concat([x[n:, :, :, :], x[:n, :, :, :]], axis=0)

        n = tf.random_uniform(
            [1], minval=0, maxval=maxval, dtype=tf.int32)[0]
        x = tf.concat([x[:, n:, :, :], x[:, :n, :, :]], axis=1)

        n = tf.random_uniform(
            [1], minval=0, maxval=maxval, dtype=tf.int32)[0]
        x = tf.concat([x[:, :, n:, :], x[:, :, :n, :]], axis=2)

        x.set_shape(self.shape)

        return x

    def _rotate_cell_and_grid(self, cell, grid):
        # This routine should be checked.

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

        cell = tf.stack([
                    cell[i],
                    cell[j],
                    cell[k]
               ])

        grid = tf.transpose(grid, pos1)
        grid.set_shape(self.shape)

        return (cell, grid)

    def _parse_cell_and_grid(self, cell, grid):
        grid = self._read_grid(grid)
        grid = self._normalize_grid(grid)

        if self.invert:
            grid = self._invert_grid(grid)

        if self.move:
            grid = self._move_grid(grid)

        # tf.py_func to use pure python function.
        cell = tf.py_func(self._read_and_nomalize_cell, [cell], tf.float32)

        if self.rotate:
            cell, grid = self._rotate_cell_and_grid(cell, grid)

        return (cell, grid)

    def _build_dataset(self):
        # Build dataset.
        cell_list = glob.glob("{}/*.grid".format(self.path))
        grid_list = [x + "data" for x in cell_list]

        cell_set = tf.data.Dataset.from_tensor_slices(cell_list)
        grid_set = tf.data.Dataset.from_tensor_slices(grid_list)

        dataset = tf.data.Dataset.zip((cell_set, grid_set)).repeat()

        if self.shuffle_size:
            dataset = dataset.shuffle(self.shuffle_size)

        dataset = dataset.map(self._parse_cell_and_grid, num_parallel_calls=1)

        if self.prefetch_size:
            dataset = dataset.prefetch(self.prefetch_size)

        self.dataset = dataset

    def write_visit_sample(self, *, cell, grid, stem, save_dir="."):
        # Inverse normalize cell.
        cmin, cmax = self.cell_length_scale
        cell = (cmax-cmin)*cell + cmin

        # Inverse normalize grid.
        lower, upper = self.energy_scale

        grid = np.array(grid.reshape([-1]))

        if self.invert:
            grid = 1.0 - grid

        grid = (upper-lower)*grid + lower

        # Make file name.
        bov = "{}/{}.bov".format(save_dir, stem)
        times = stem + ".times"

        size = self.shape[0]
        # Write header file.
        with open(bov, "w") as bovfile:
            bovfile.write(textwrap.dedent("""\
                TIME: 1.000000
                DATA_FILE: {}
                DATA_SIZE:     {size} {size} {size}
                DATA_FORMAT: FLOAT
                VARIABLE: data
                DATA_ENDIAN: LITTLE
                CENTERING: nodal
                BRICK_ORIGIN:        0  0  0
                BRICK_SIZE:       {} {} {}""".format(
                times, size=size, *cell)
            ))
        # Write times file.
        grid.tofile("{}/{}".format(save_dir, times))

    def write_sample(self, *, x, stem, save_dir="."):
        raise Exception("Do not use")

        lower, upper = self.energy_scale

        x = np.array(x.reshape([-1]))

        if self.invert:
            x = 1.0 - x

        x = (upper-lower)*x + lower
        # Write times file.
        x.tofile("{}/{}.{}".format(save_dir, stem. self.extension))


if __name__ == "__main__":
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    egrid_dataset = EnergyGridDataset(
                        path="/home/FRAC32/IZA_CUBIC",
                        shape=32,
                        invert=True,
                        rotate=True,
                        move=True,
                        extension="griddata",
                        energy_limit=[-4000, 5000],
                        energy_scale=[-4000, 5000],
                        shuffle_size=10,
                        prefetch_size=10,
                    )

    iterator = egrid_dataset.dataset.batch(10).make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        grids = sess.run(iterator.get_next())

        for i, grid in enumerate(grids):
            egrid_dataset.write_visit_sample(
                x=grid,
                stem="sample_{}".format(i),
                save_dir=".",
            )
    """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    egrid_dataset = EnergyGridTupleDataset(
                        #path="/home/FRAC32/IZA_CUBIC",
                        path=".",
                        shape=32,
                        invert=True,
                        rotate=True,
                        move=False,
                        energy_limit=[-4000, 5000],
                        energy_scale=[-4000, 5000],
                        cell_length_scale=[0.0, 60.0],
                        shuffle_size=10,
                        prefetch_size=10,
                    )

    iterator = egrid_dataset.dataset.batch(10).make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        cells, grids = sess.run(iterator.get_next())

        for i, (cell, grid) in enumerate(zip(cells, grids)):
            egrid_dataset.write_visit_sample(
                cell=cell,
                grid=grid,
                stem="sample_{}".format(i),
                save_dir=".",
            )
