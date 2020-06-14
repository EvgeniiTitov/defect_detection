from app.visual_detector.neural_networks.wood_classifier.nn import Model
from PIL import Image
from typing import List
import torch.nn.functional as F
import torch
import numpy as np
import cv2


class WoodCrackSegmenter:
    path_to_weights = r"D:\Desktop\Reserve_NNs\weights_configs\defect_detectors\wood_crack\without_crosses.hdf5"
    background_colour = (255, 255, 255)
    IMG_HEIGHT, IMG_WIDTH = 896, 224

    def __init__(self):
        try:
            self._model = Model(
                self.IMG_HEIGHT,
                self.IMG_WIDTH,
                self.path_to_weights
            )
        except Exception as e:
            print(f"\nFailed during WoodCrackDetector initialization. Error: {e}")
            raise e

    def predict_batch_torch_tensor(self, images_on_gpu: List[torch.Tensor]):
        """

        :param images_on_gpu:
        :return:
        """
        # TODO: Check if torch.Tensor can be supplied

        towers_to_remove_background = list()
        for image in images_on_gpu:
            # Resize each image to the expected size
            try:
                resized_image = F.interpolate(
                    image.unsqueeze_(0),
                    size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                    mode="nearest"
                )
            except Exception as e:
                print(f"Failed during wooden tower resizing. Error: {e}")
                raise e
            towers_to_remove_background.append(resized_image)

        # Concatinate towers into one tensor
        try:
            batch = torch.cat(towers_to_remove_background)
        except Exception as e:
            print(f"Failed during wooden tower concatination. Error: {e}")
            raise e

        # Get predictions
        #masks = self._model.predict(batch)

    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Receives a list with images of wooden towers
        """
        # Check if image's width > height, then cut the image in two to ensure each wooden
        # pole is a separate image.
        output = {i: [] for i in range(len(images))}
        images_to_process = list()
        for image in images:
            height, width = image.shape[:2]
            if width > height:
                image_left = image[0: height, 0: int(width / 2)]
                image_right = image[0: height, int(width / 2) + 1: width]
                images_to_process.append(image_left)
                images_to_process.append(image_right)
                continue
            images_to_process.append(image)

        return self.predict_batch_np_array(images_to_process)

    def predict_batch_np_array(self, images_on_cpu: List[np.ndarray]) -> List[np.ndarray]:
        """

        height, width, channel
        :param images_on_cpu:
        :return:
        """
        # Preprocess images by normalizing and resizing them
        towers_to_remove_background = list()
        for image in images_on_cpu:
            try:
                image_resized = cv2.resize(
                    image,
                    (self.IMG_WIDTH, self.IMG_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )
                normalized_image = np.array([image_resized / 255.0])
            except Exception as e:
                print(f"Failed during image preprocessing in WoodCrackSegment. Error: {e}")
                continue
            towers_to_remove_background.append(normalized_image)

        # Concatenate images into one array
        try:
            batch_images = np.concatenate(towers_to_remove_background)
        except Exception as e:
            print(f"Failed during np.ndarray concatination in WoodCrackSegment. Error: {e}")
            raise e

        # Run inference to get masks
        masks = self._model.predict(batch_images)

        # Postprocess results
        assert len(masks) == len(images_on_cpu), "Nb of images != nb of detected masks"
        masked_images = list()
        for i in range(len(masks)):
            masked_image = self.postprocess_results(images_on_cpu[i], masks[i])
            masked_images.append(masked_image)

        return masked_images

    def postprocess_results(self, image: np.ndarray, mask: list) -> np.ndarray:
        """

        :param image:
        :param mask:
        :return:
        """
        # Binarize image
        binarized_image = self.binarize_mask(mask)

        # Fill background colour
        output_image = self.fill_background(image, binarized_image)

        return output_image

    def binarize_mask(self, mask):
        """

        :param mask:
        :return:
        """
        mask = mask.squeeze()
        denormalize_item_vec = np.vectorize(lambda i: 255 if i > 0.95 else 0)
        mask = denormalize_item_vec(mask)
        mask = np.asarray([np.array(r, dtype=np.uint8) for r in mask])

        return mask

    def fill_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """

        :param image:
        :param mask:
        :return:
        """
        mask = Image.fromarray(mask).convert('L')  # black n white
        # Resize image to the mask's size
        resized_image = cv2.resize(image, mask.size)  # (width, height)
        masked_image = self.apply_mask(np.asarray(resized_image), np.asarray(mask))

        return cv2.resize(masked_image, (image.shape[1], image.shape[0]))

    def apply_mask(self, img: np.ndarray, mask: Image):
        result = []
        for i in range(len(mask) - 1):
            row = []
            for j in range(len(mask[0]) - 1):
                row.append(
                    np.array(self.background_colour, dtype=np.uint8) if mask[i][j] != 255 else img[i][j]
                )
            result.append(row)

        return self.spikes_to_img(result)

    def spikes_to_img(self, spikes: list) -> np.ndarray:
        arr = np.asarray([np.array(r, dtype=np.uint8) for r in spikes])

        return arr
