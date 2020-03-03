from tensorflow.keras import layers
from grid import make_grid, apply_homography, bilinear_sampler


class Homography(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):

        if type(inputs) is not list or len(inputs) != 2:
            raise Exception(
                "Homography must be called on a list of 2 tensors, image and homography. Got: "
                + str(inputs)
            )
        images = inputs[0]
        grids = make_grid(*images.shape)
        homos = inputs[1]
        transformed_grids = apply_homography(homos, grids)
        return bilinear_sampler(images, transformed_grids)


if __name__ == "__main__":
    
    import numpy as np
    hom_layer = Homography()
    id_transform = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], "float32"
    )
    homo_layer = Homography()
    x_train = np.ones((1, 8, 8))
    out = homo_layer([x_train, id_transform])
    print(out)
