import tensorflow as tf

def make_grid(batch_size, width, height):
    """
    Create batch_size grids of unit-squares
    >>> make_grid(4,28,28).shape
    TensorShape([4, 28, 28, 2])
    """

    xs = tf.linspace(0., float(width) - 1., width)
    xs = tf.cast(xs, "int32")

    ys = tf.linspace(0., float(width) - 1., height)
    ys = tf.cast(ys, "int32")

    xs, ys = tf.meshgrid(xs, ys)

    grid = tf.stack([ys, xs], 2)
    grid = tf.expand_dims(grid, 0)

    batch_grids = tf.tile(grid, [batch_size, 1, 1, 1])
    return batch_grids


if __name__ == "__main__":
    print(make_grid(8, 8, 8))
