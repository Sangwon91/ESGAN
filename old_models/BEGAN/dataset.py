import tensorflow as tf

def make_energy_grid_dataset(*,
        path,
        shape,
        batch_size,
        energy_range,
        shuffle_size=None,
        prefetch_size=None,
        shared_name=None,
        invert=True):

    if type(shape) not in (list, tuple):
        shape = [shape, shape, shape, 1]

    min_energy = energy_range[0]
    max_energy = energy_range[1]

    def _filename_parser(x):
        # Load data from the file.
        x = tf.read_file(x)
        x = tf.decode_raw(x, out_type=tf.float32)

        # reshape to 3D with 1 channel (total 4D).
        x = tf.reshape(x, shape)

        # Drop NaN in array.
        # if x[...] is NaN, it becames max_energy.
        x = tf.where(tf.is_nan(x), max_energy*tf.ones_like(x), x)

        # Normalize.
        x = tf.clip_by_value(x, min_energy, max_energy)
        x = (x - min_energy) / (max_energy - min_energy)

        # Maximum energy to zero.
        if invert:
            x = 1.0 - x

        return x


    with tf.variable_scope("dataset"):
        # Build dataset.
        voxel_path_expression = "{}/*.voxel".format(path)

        dataset = tf.data.Dataset.list_files(voxel_path_expression)
        dataset = dataset.repeat()

        if shuffle_size:
            dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.map(_filename_parser, num_parallel_calls=1)

        if prefetch_size:
            dataset = dataset.prefetch(prefetch_size)

        dataset = dataset.batch(batch_size)

    return dataset





