import functools
import tensorflow as tf

def sigmoid_log_with_logits(x):
    """Equivalent to log(sigmoid(x)) but numerically safe."""
    with tf.variable_scope("sigmoid_log_with_logits"):
        y = tf.minimum(x, 0.0) - tf.log(1.0 + tf.exp(-tf.abs(x)))

    return y


kernel_initializer = tf.random_normal_initializer(0.0, 0.02)


dense = functools.partial(
    tf.layers.dense,
    activation=None,
    use_bias=False,
    kernel_initializer=kernel_initializer,
)


conv3d = functools.partial(
    tf.layers.conv3d,
    kernel_size=4,
    strides=2,
    padding="SAME",
    activation=None,
    use_bias=False,
    kernel_initializer=kernel_initializer,
)


conv3d_transpose = functools.partial(
    tf.layers.conv3d_transpose,
    kernel_size=5,
    strides=2,
    padding="SAME",
    activation=None,
    use_bias=False,
    kernel_initializer=kernel_initializer,
)

# Source: https://github.com/maxorange/voxel-dcgan/blob/master/ops.py
# Automatic updator version of batch normalization.
def batch_normalization(
        x, training, name="batch_normalization", decay=0.99, epsilon=1e-5):
    train = training

    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', shape[-1:],
            initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02))
        moving_mean = tf.get_variable('moving_mean', shape[-1:],
            initializer=tf.constant_initializer(0.), trainable=False)
        moving_var = tf.get_variable('moving_var', shape[-1:],
            initializer=tf.constant_initializer(1.), trainable=False)

        if moving_mean not in tf.moving_average_variables():
            collection = tf.GraphKeys.MOVING_AVERAGE_VARIABLES
            tf.add_to_collection(collection, moving_mean)
            tf.add_to_collection(collection, moving_var)

        def train_mode():
            # execute at training time
            axes = list(range(len(shape) - 1))
            # It really works?
            batch_mean, batch_var = tf.nn.moments(x, axes=axes)
            update_mean = tf.assign_sub(
                moving_mean, (1-decay) * (moving_mean-batch_mean)
            )
            update_var = tf.assign_sub(
                moving_var, (1-decay) * (moving_var- batch_var)
            )

            # Automatically update global means and variances.
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(
                            x, batch_mean, batch_var, beta, gamma, epsilon)

        def test_mode():
            # execute at test time
            return tf.nn.batch_normalization(
                       x, moving_mean, moving_var, beta, gamma, epsilon)

        return tf.cond(train, train_mode, test_mode)


def minibatch_discrimination(x, num_kernels, dim_per_kernel, name="minibatch"):
    input_x = x

    with tf.variable_scope(name):
        x = dense(x, units=num_kernels*dim_per_kernel)
        x = tf.reshape(x, [-1, num_kernels, dim_per_kernel])

        diffs = (
            tf.expand_dims(x, axis=-1) -
            tf.expand_dims(tf.transpose(x, [1, 2, 0]), axis=0)
        )

        l1_dists = tf.reduce_sum(tf.abs(diffs), axis=2)
        minibatch_features = tf.reduce_sum(tf.exp(-l1_dists), axis=2)

        return tf.concat([input_x, minibatch_features], axis=1)
