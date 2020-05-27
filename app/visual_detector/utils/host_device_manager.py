from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple
import cv2
import os
import numpy as np
import torch
import torchvision


class HostDeviceManager:

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
                image_tensor = HostDeviceManager.preprocess_image_including_resizing(image, img_size)
            else:
                image_tensor = HostDeviceManager.preprocess_image(image)

            image_tensor.unsqueeze_(0)
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
            print("Images successfully moved from host to device")
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

        # TODO: Check if you need Normalize() here for your system

        image_transforms = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )]
        )
        return image_transforms(image)

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
    def resize_tensor(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
        """
        Receives torch tensor and resizes it.

        F.interpolate - Down/up samples the input to either the given size or the given scale_factor
        :param tensor:
        :param new_size:
        :return:
        """
        assert new_size > 0, "Cannot resize tensor to a negative value"
        resized_tensor = F.interpolate(input=tensor, size=(new_size, new_size))

        return resized_tensor

    @staticmethod
    def resize_tensor_v2(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """
        Resizes, but does not keep the aspect ratio.
        :param tensor:
        :param size:
        :return:
        """
        return (F.adaptive_avg_pool2d(tensor, size)).data

    @staticmethod
    def resize_tensor_keep_aspratio(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
        """

        :param tensor:
        :param new_size:
        :return:
        """
        tensor_height, tensor_width = tensor.shape[:2]

        # Before resizing tensor to 416 x 416 keeping the aspect ratio, determine which side is greater
        tall = False
        if tensor_height > tensor_width:
            tall = True

        # Calculate the coefficient to keep aspect ratio
        if tall:
            coef = new_size / float(tensor_height)
        else:
            coef = new_size / float(tensor_width)

        # Calculate size of the other side


        # Resize to the new sizes


        # Create tensor 416 by 416, and put your new image in it




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
    def visualise_sliced_img(images: List[torch.Tensor]) -> None:
        for image in images:
            image = image.permute(1, 2, 0)
            # TODO: Check how to drop torch.Size([416, 416, 3, 5]) batch size at the end?
            image = image.cpu().numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # plt.imshow(image)
            cv2.imshow("window", image_rgb)
            cv2.waitKey(0)