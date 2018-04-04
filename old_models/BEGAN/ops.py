import functools
import tensorflow as tf

def up_sampling3d(tensor, scale, name="up_sampling3d"):
    """
    Args:
        tensor: input tensor of 5 dimension [batch, dim1, dim2, dim3, channels]
        scale: scale factor, can be an integer or tuple/list of 3 integers.
    """
    if isinstance(scale, int):
        repeats = [1, scale, scale, scale, 1]
    elif type(scale) in (list, tuple):
        assert len(scale) == 3
        repeats = [1] + scale + [1]
    else:
        raise ValueError("use integer or list of intergers for scale")

    multiples = [1] + repeats

    with tf.variable_scope(name):
        # Same as numpy.repeat()
        x = tf.expand_dims(tensor, axis=-1)
        x = tf.tile(x, multiples=multiples)
        # Does not work for 3D data.
        # x = tf.reshape(x, tf.shape(tensor) * repeats)
        np_shape = [i if i else -1 for i in tensor.get_shape().as_list()]
        shape = [s*r for s, r in zip(np_shape, repeats)]

        x = tf.reshape(x, shape)

    return x


def down_sampling3d(tensor, scale, name="down_sampling3d"):
    """
    Args:
        tensor: input tensor of 5 dimension [batch, dim1, dim2, dim3, channels]
        scale: scale factor, can be an integer or tuple/list of 3 integers.
    """

    with tf.variable_scope(name):
        x = tf.layers.average_pooling3d(
                inputs=tensor,
                pool_size=scale,
                strides=scale,
                padding="SAME")

    return x


def is_same_shape(tensor1, tensor2, neglect_none=True):
    list1 = tensor1.get_shape().as_list()
    list2 = tensor2.get_shape().as_list()

    if not neglect_none:
        if None in list1 or None in list2:
            return False

    return list1 == list2


def vanishing_residual_layer(x, next_x, vanishing_residual, carry, name=None):
    if not name:
        name = "vanishing_residual"

    with tf.variable_scope(name):
        def true_fn():
            with tf.variable_scope("vanishing"):
                return carry*x + (1.0-carry)*next_x

        def false_fn():
            with tf.variable_scope("no_vanishing"):
                return tf.identity(next_x)

        result = tf.cond(
            vanishing_residual,
            true_fn=true_fn,
            false_fn=false_fn
        )

    return result


conv3d = functools.partial(
    tf.layers.conv3d,
    kernel_size=3,
    strides=1,
    padding="SAME",
    activation=tf.nn.elu,
    use_bias=False,
)


def test():
    a = tf.placeholder(shape=[None, 2, 4], dtype=tf.float32)
    b = tf.placeholder(shape=[None, 2, 4], dtype=tf.float32)

    print(is_same_shape(a, b, neglect_none=True), True)
    print(is_same_shape(a, b, neglect_none=False), False)

    import numpy as np

    v = np.random.uniform(0.0, 1.0, size=[4, 7, 13, 4, 2]).astype(np.float32)

    v_np = v
    for axis in range(1, 4):
        v_np = np.repeat(v_np, 3, axis=axis)

    v_up = up_sampling3d(tf.Variable(v), 3)
    v_down = down_sampling3d(tf.Variable(v_np), 3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v_up_val, v_down_val = sess.run([v_up, v_down])

    print(np.allclose(v_np, v_up_val))
    print(np.allclose(v, v_down_val))

if __name__ == "__main__":
    test()
