import segmentation_models as sm
from typing import List
import numpy as np


class Model:

    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width
        self. _BACKBONE = 'resnet34'
        self._LR = 0.0001
        self._IMG_CHANNELS = 3
        self._EPOCHS = 24
        self._model = None

    def compile(self) -> None:
        try:
            self._model = sm.Unet(
                self._BACKBONE,
                input_shape=(self.image_height, self.image_width, self._IMG_CHANNELS),
                classes=1,
                activation='sigmoid'
            )
            # optim = keras.optimizers.Adam(self._LR)
            # total_loss = sm.losses.binary_focal_dice_loss
            # metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
            # self._model.compile(optim, total_loss, metrics)
        except Exception as e:
            print(f"Failed during model compilation. Error: {e}")
            raise e

    def load_weights(self, weights_path: str) -> None:
        try:
            self._model.load_weights(weights_path)
        except Exception as e:
            print(f"Failed during weights loading. Error: {e}")
            raise e

    def predict(self, images: List[np.ndarray]):
        # In case multiple images sent for inference, concat them into a single array
        try:
            #images = np.asarray(images)  # .squeeze()
            images = np.concatenate(images)
        except Exception as e:
            print(f"Failed during np.ndarray concatination in WoodCrackSeg. Error: {e}")
            raise e

        return self._model.predict(images, batch_size=len(images))
