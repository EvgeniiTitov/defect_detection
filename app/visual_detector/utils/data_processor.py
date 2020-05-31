from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
import cv2
import os
import numpy as np
import torch
import torchvision
import sys


class DataProcessor:

    @staticmethod
    def load_images_to_GPU(images: List[np.ndarray], img_size=None) -> torch.Tensor:
        """
        Loads images on GPU (!) without resizing them.
        :param images:
        :param img_size:
        :return:
        """
        image_tensors = list()
        for image in images:
            # Preprocess images before .cat()ing them.
            if img_size:
                print("ATTENTION: Images will be resized before being moved to GPU")
                image_tensor = DataProcessor.preprocess_image_including_resizing(image, img_size)
            else:
                image_tensor = DataProcessor.preprocess_image(image)
            #HostDeviceManager.visualise_sliced_img(image_tensor, "fromOptimizer")
            image_tensors.append(image_tensor)

        # Concat torch tensors into one tensor
        try:
            batch = torch.cat(image_tensors)
        except Exception as e:
            print(f"Failed during .cat()ing torch tensors. Error: {e}")
            raise

        # Move the new tensor to GPU and return reference to it
        try:
            batch_gpu = batch.cuda()
            return batch_gpu
        except Exception as e:
            print(f"\nATTENTION: Moving images to GPU failed. Error: {e}")
            return batch

    @staticmethod
    def preprocess_image(image: np.ndarray) -> torch.Tensor:
        """
        :param image:
        :return:
        """
        img = image[:, :, ::-1].transpose((2, 0, 1)).copy()  # rgb -> bgr, change channel order
        img = torch.from_numpy(img).float().unsqueeze(0)

        return img

    @staticmethod
    def preprocess_image_including_resizing(image: np.ndarray, new_size: int) -> torch.Tensor:
        image_transforms = torchvision.transforms.Compose([
            transforms.Resize((new_size, new_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )]
        )
        # Cast image to PIL's Image since .Resize() and .Crop() do not work with np.ndarrays
        try:
            image_pil = Image.fromarray(image)
        except Exception as e:
            print(f"Failed during converting np.ndarray -> to PIL's Image. Error: {e}")
            raise

        return image_transforms(image_pil)

    @staticmethod
    def resize_tensor_keeping_aspratio(
            batch_tensor: torch.Tensor,
            new_size: int,
            background_color: float = 128.
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Receives torch tensor (batch of images) and resizes it keeping the aspect ratio and filling
        the rest of the image
        :param batch_tensor:
        :param new_size:
        :param background_color:
        :return:
        """
        assert new_size > 0, "Cannot resize tensor to a negative value"
        tensor_height, tensor_width = batch_tensor.shape[2:4]
        batch_size = batch_tensor.shape[0]
        # Before resizing tensor to new_size keeping the aspect ratio, determine which side is greater
        tall = True if tensor_height >= tensor_width else False
        # Calculate the coefficient to keep aspect ratio
        coef = new_size / float(tensor_height) if tall else new_size / float(tensor_width)
        # Resize to the new sizes
        new_dimension = [new_size, int(tensor_width * coef)] if tall else [int(tensor_height * coef), new_size]

        # TODO: Implement a proper fix for the following:
        # There can be a situation where new dim [0] or [1] is not even which doesn't allow to pick the padding
        # number easily. If this is the case, make sure both new dim values are even.
        if new_dimension[0] % 2 != 0:
            new_dimension[0] += 1
        elif new_dimension[1] % 2 != 0:
            new_dimension[1] += 1

        # Resize tensor to a new size (width and height)
        resized_tensor = F.interpolate(batch_tensor, new_dimension)
        # Create a new tensor of the required output shape filled with grey colour
        res = torch.ones(batch_size, 3, new_size, new_size).cuda() * background_color
        # Calculate padding size
        padding = (new_size - new_dimension[1])//2 if tall else (new_size - new_dimension[0])//2
        if tall:
            res[:, :, :, padding:new_size - padding] = resized_tensor
        else:
            res[:, :, padding:new_size - padding, :] = resized_tensor

        return res, (tensor_height, tensor_width)

    @staticmethod
    def slice_out_tensor(image: torch.Tensor, coordinates: list) -> torch.Tensor:
        """
        Slices out a tensor using the coordinates provided
        :param image:
        :param coordinates:
        :return:
        """
        assert isinstance(image, torch.Tensor), "Wrong image data type. Tensor expected"
        assert len(coordinates) == 4, "No or wrong number of coordinates provided. Expected 4"
        left = coordinates[0]
        top = coordinates[1]
        right = coordinates[2]
        bot = coordinates[3]
        try:
            subimage = image[:, top:bot, left:right]
        except Exception as e:
            print(f"Failed while slicing out a tensor. Error: {e}")
            raise e

        return subimage

    @staticmethod
    def recalculate_bb(scaling_factor: float, detections: dict) -> dict:
        """
        NOTE: Doesn't work as intended
        :param scaling_factor:
        :param detections:
        :return:
        """
        output = dict()
        for img_batch_index, detections in detections.items():
            output[img_batch_index] = list()
            # recalculate bb coordinates
            for detection in detections:
                detection = detection[0]
                #TODO: Potential error if top or left = 0
                new_left = int(detection[0] / scaling_factor)
                new_top = int(detection[1] / scaling_factor)
                new_right = int(detection[2] / scaling_factor)
                new_bot = int(detection[3] / scaling_factor)
                obj_score = detection[4]
                conf = detection[5]
                index = detection[6]
                output[img_batch_index].append(
                    [new_left, new_top, new_right, new_bot, obj_score, conf, index]
                )
        return output

    @staticmethod
    def rescale_bb(
            detections: dict,
            current_dim: int,
            original_shape: tuple = None,
            original_shapes: List[tuple] = None
    ) -> dict:
        """
        Rescale bounding boxes relatively to size(s) of the original image(s)
        :param detections:
        :param current_dim:
        :param original_shape:
        :param original_shapes:
        :return:
        """
        msg = "ERROR: Expected either ORIGINAL_SHAPE if all imgs were of the same size before resizing " \
              "or ORIGINAL_SHAPES if resized images varied in size. Both or none seem to have been provided!"
        assert any ((original_shape, original_shapes)) and not all((original_shape, original_shapes)), msg

        # If all images were of the same size before they'd been resized
        if original_shape and not original_shapes:
            original_h, original_w = original_shape
            # Added padding
            pad_x = max(original_h - original_w, 0) * (current_dim / max(original_shape))
            pad_y = max(original_w - original_h, 0) * (current_dim / max(original_shape))
            # Image height after padding's been removed
            unpad_h = current_dim - pad_y
            unpad_w = current_dim - pad_x

        # Traverse over detections for the batch and rescale bb of any detected boxes
        output = dict()
        for img_batch_index, detections in detections.items():
            output[img_batch_index] = list()

            # In case images were of different sizes before they'd been resized, get original size for
            # each image in the batch and rescale bb accordingly
            if original_shapes and not original_shape:
                payload = original_shapes[img_batch_index]
                assert payload[0] == img_batch_index, "ERROR: Hey b0ss, I have an error"
                original_h, original_w = payload[1]

                # Added padding
                pad_x = max(original_h - original_w, 0) * (current_dim / max(payload[1]))
                pad_y = max(original_w - original_h, 0) * (current_dim / max(payload[1]))
                # Image height after padding's been removed
                unpad_h = current_dim - pad_y
                unpad_w = current_dim - pad_x

            for detection in detections:
                detection = detection[0]
                # Rescale boxes
                new_left = int(((detection[0] - pad_x // 2) / unpad_w) * original_w)

                # TODO: Find error here (https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/utils.py)
                new_top = int(((detection[1] - pad_y // 2) / unpad_h) * original_h)
                new_top = 1 if new_top <= 0 else new_top

                new_right = int(((detection[2] - pad_x // 2) / unpad_w) * original_w)

                new_bot = int(((detection[3] - pad_y // 2) / unpad_h) * original_h)
                new_bot = int(original_h * 0.999) if new_bot >= original_h else new_bot

                obj_score = round(detection[4], 4)
                conf = round(detection[5], 4)
                index = int(detection[6])
                # Save modified results
                output[img_batch_index].append(
                    [new_left, new_top, new_right, new_bot, obj_score, conf, index]
                )
        return output

    @staticmethod
    def modify_bb_coord(tower, image: torch.Tensor, nb_of_towers: int) -> None:
        """
        Modifies tower's bb depending on the total nb of towers detected on the image.
        TODO: Overlapping check, currently widen boxes might overlap. As a result, if some component
              happens to be inside the overlapping area, it will be within 2 bb of 2 different towers,
              hence it will get detected and checked for defects twice
        :param tower: detected tower
        :param image: image on which the tower was detected
        :return:
        """
        img_width, img_height = image.shape[1:3]
        if nb_of_towers == 1:
            new_left_boundary = int(tower.BB_left * 0.4)
            new_right_boundary = int(tower.BB_right * 1.6) if int(tower.BB_right * 1.6) < \
                                                                    img_width else (img_width - 2)
            # Move upper border way up, often when a pole is close up many components do not get
            # included in the box, as a result they do not get found
            new_top_boundary = int(tower.BB_top * 0.1)
            new_bot_boundary = int(tower.BB_bottom * 1.1) if int(tower.BB_bottom * 1.1) < \
                                                                    img_height else (img_height - 2)
        else:
            new_left_boundary = int(tower.BB_left * 0.9)
            new_right_boundary = int(tower.BB_right * 1.1) if int(tower.BB_right * 1.1) < \
                                                                    img_width else (img_width - 2)
            new_top_boundary = int(tower.BB_top * 0.5)
            new_bot_boundary = int(tower.BB_bottom * 1.1) if int(tower.BB_bottom * 1.1) < \
                                                                    img_height else (img_height - 2)
        try:
            tower.update_object_coordinates(
                left=new_left_boundary,
                top=new_top_boundary,
                right=new_right_boundary,
                bottom=new_bot_boundary
            )
        except Exception as e:
            print(f"Failed during BB coord updating. Error: {e}")
            raise e

    @staticmethod
    def read_images(paths: list) -> list:
        """
        Opens images using PIL.Image
        :param paths:
        :return:
        """
        images = []
        for path_to_image in paths:
            try:
                # image = Image.open(path_to_image)
                image = cv2.imread(path_to_image)
            except Exception as e:
                print(f"Failed to open image {path_to_image}. Error: {e}")
                continue
            images.append(image)

        return images

    @staticmethod
    def visualise_sliced_img(images: torch.Tensor, name=None) -> None:
        for i, image in enumerate(images):
            image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            save_path = r"D:\Desktop\system_output\OUTPUT"
            cv2.imwrite(os.path.join(save_path, f"{name}_{i}.jpg"), image)
            # cv2.imshow("window", image)
            # cv2.waitKey(0)

    @staticmethod
    def show_image(image: np.ndarray) -> None:
        cv2.imshow("", image)
        cv2.waitKey(0)
