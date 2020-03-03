from tensorflow.keras import layers


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
        transformed_grids =  apply_homogrpahy(homos,grids)
        return bilinear_sampler(images,transformed_grids)
        
