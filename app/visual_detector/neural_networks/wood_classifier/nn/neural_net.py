import segmentation_models as sm
import tensorflow as tf
import keras
import numpy as np


class Model:

    def __init__(
            self,
            image_height: int,
            image_width: int,
            path_to_weights: str
    ):
        self.image_height = image_height
        self.image_width = image_width
        self. _BACKBONE = 'resnet34'
        self._LR = 0.0001
        self._IMG_CHANNELS = 3

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

        self._model = None
        self.compile(path_to_weights)

    def compile(self, path_to_weights: str) -> None:
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    self._model = sm.Unet(
                        self._BACKBONE,
                        input_shape=(self.image_height, self.image_width, self._IMG_CHANNELS),
                        classes=1,
                        activation='sigmoid'
                    )
                    self._model.compile(
                        keras.optimizers.Adam(self._LR),
                        sm.losses.binary_focal_dice_loss,
                        [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
                    )
                    self._model.load_weights(path_to_weights)
                except Exception as e:
                    print(f"Failed during model initialization and compilation. Error: {e}")
                    raise e

    def predict(self, images: np.ndarray):
        with self.graph.as_default():
            with self.session.as_default():
                prediction = self._model.predict(images, batch_size=len(images))

        return prediction
