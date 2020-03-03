import tensorflow as tf


def make_grid(batch_size, width, height):
    """
    Create batch_size grids of unit-squares
    >>> make_grid(4,28,28).shape
    TensorShape([4, 28, 28, 2])
    """

    xs = tf.linspace(0.0, float(width) - 1.0, width)
    xs = tf.cast(xs, "int32")

    ys = tf.linspace(0.0, float(width) - 1.0, height)
    ys = tf.cast(ys, "int32")

    xs, ys = tf.meshgrid(xs, ys)

    grid = tf.stack([ys, xs], 2)
    grid = tf.expand_dims(grid, 0)

    batch_grids = tf.tile(grid, [batch_size, 1, 1, 1])
    return tf.cast(batch_grids,"float32")


def corners(grid):
    """
    For a given grid, return the corner coordinates and offsets
    >>> crnrs,dlts = corners(make_grid(1,1,1))
    >>> crnrs.numpy()
    array([[[[[0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]]], dtype=int32)
    >>> dlts.numpy()
    array([[[[1., 0., 0., 0.]]]], dtype=float32)
    """

    batch_size, width, height, dim = grid.shape
    new_shape = [-1]
    coords = tf.reshape(grid, [batch_size * width * height, dim])
    coords = tf.cast(coords, "float32")

    xs = coords[:, 0]
    ys = coords[:, 1]

    xs0 = tf.cast(tf.floor(xs), "int32")
    xs1 = xs0 + 1
    ys0 = tf.cast(tf.floor(ys), "int32")
    ys1 = ys0 + 1

    # calculate deltas
    xs0 = tf.cast(xs0, "float32")
    xs1 = tf.cast(xs1, "float32")
    ys0 = tf.cast(ys0, "float32")
    ys1 = tf.cast(ys1, "float32")

    wa = (xs1 - xs) * (ys1 - ys)
    wb = (xs1 - xs) * (ys - ys0)
    wc = (xs - xs0) * (ys1 - ys)
    wd = (xs - xs0) * (ys - ys0)

    deltas = tf.stack([wa, wb, wc, wd], 1)
    deltas = tf.reshape(deltas, [batch_size, width, height, 4])

    # clipping, to image boundaries
    # essentially , we are taking the border values and extend
    xs0 = tf.clip_by_value(xs0, 0, width - 1)
    xs1 = tf.clip_by_value(xs1, 0, width - 1)
    ys0 = tf.clip_by_value(ys0, 0, height - 1)
    ys1 = tf.clip_by_value(ys1, 0, height - 1)

    ul = tf.stack([xs0, ys0], 1)
    ur = tf.stack([xs1, ys0], 1)
    ll = tf.stack([xs0, ys1], 1)
    lr = tf.stack([xs1, ys1], 1)
    corners_grid = tf.stack([ul, ur, ll, lr], 1)
    corners_grid = tf.cast(corners_grid, "int32")

    return tf.reshape(corners_grid, [batch_size, width, height, 4, 2]), deltas


def bilinear_sampler(images, grid):
    """
    >>> import numpy as np
    >>> img = np.random.rand(1,2,2)
    >>> grid = make_grid(1,2,2)
    >>> smpld = bilinear_sampler(img,grid)
    >>> np.allclose(img,smpld)
    True
    """
    corners_grid, deltas = corners(
        grid
    )  # batch size, width, heigth, 4 corners, 2 dimensional

    corner_values = tf.gather_nd(images, corners_grid, batch_dims=1)
    corner_values = tf.cast(corner_values, "float32")
    corner_values = tf.reshape(corner_values, deltas.shape)

    interpolated = tf.math.multiply(corner_values, deltas)
    return tf.math.reduce_sum(interpolated, axis=3)


def apply_homogrpahy(homo, grid):
    """
    Applies a transformation matrix in projective space to grid
    >>> id_transform = np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
    >>> grid = tf.cast(make_grid(1,2,2),"float64")
    >>> np.array_equal(apply_homogrpahy(id_transform,grid), grid)
    True
    """
    # compute flat projective coordinates
    batch_size = grid.shape[0]
    flat_grid = tf.reshape(grid, [batch_size, -1, 2])
    ones = tf.ones(flat_grid.shape[: (len(flat_grid.shape) - 1)], "float32")
    flat_grid_projective = tf.stack([flat_grid[:, :, 0], flat_grid[:, :, 1], ones], 2)
    # apply the transform
    transformed = tf.matmul(flat_grid_projective, homo)
    # map back to cartesian coordinates
    transformed = transformed[:, :, :2] / transformed[:, :, 2:]
    return tf.reshape(transformed, grid.shape)


if __name__ == "__main__":
    import doctest
    import numpy as np

    doctest.testmod(verbose=True, extraglobs={"np": np, "tf": tf})
