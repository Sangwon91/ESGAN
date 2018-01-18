import random
import tensorflow as tf

def make_energy_grid_dataset(*,
    path,
    shape,
    #batch_size,
    invert=True,
    rotate=True,
    move=True,
    extension="voxel",
    energy_limit=[-5000.0, 5000.0],
    energy_scale=[-6000.0, 6000.0],
    shuffle_size=None,
    prefetch_size=None,
    shared_name=None):

    if type(shape) not in (list, tuple):
        shape = [shape, shape, shape, 1]

    def _filename_parser(x):
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

        """
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
        """

        x.set_shape(shape)
        return x


    with tf.variable_scope("dataset"):
        # Build dataset.
        voxel_path_expression = "{}/*.{}".format(path, extension)

        dataset = tf.data.Dataset.list_files(voxel_path_expression)
        dataset = dataset.repeat()

        if shuffle_size:
            dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.map(_filename_parser, num_parallel_calls=2)

        if prefetch_size:
            dataset = dataset.prefetch(prefetch_size)

        #dataset = dataset.batch(batch_size)

    return dataset
