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
                print("ATTENTION: Images will be resized before being moved to GPU")
                image_tensor = HostDeviceManager.preprocess_image_including_resizing(image, img_size)
            else:
                image_tensor = HostDeviceManager.preprocess_image(image)
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
        img = image[:, :, ::-1].transpose((2, 0, 1)).copy()  # rgb -> bgr
        img = torch.from_numpy(img).float().unsqueeze(0)  # div(255.0) here makes them almost black!

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
    def resize_tensor(tensor: torch.Tensor, new_size: int, background_color: float = 128.) -> torch.Tensor:
        """
        Receives torch tensor and resizes it keeping the aspect ratio and filling the rest of the image
        :param tensor:
        :param new_size:
        :param background_color:
        :return:
        """
        assert new_size > 0, "Cannot resize tensor to a negative value"
        tensor_height, tensor_width = tensor.shape[2:4]
        batch_size = tensor.shape[0]

        # Before resizing tensor to 416 x 416 keeping the aspect ratio, determine which side is greater
        tall = True if tensor_height >= tensor_width else False
        # Calculate the coefficient to keep aspect ratio
        coef = new_size / float(tensor_height) if tall else new_size / float(tensor_width)
        # Resize to the new sizes
        new_dimension = (new_size, int(tensor_width * coef)) if tall else (int(tensor_height * coef), new_size)
        # Resize tensor to a new size (width and height)
        resized_tensor = F.interpolate(tensor, new_dimension)
        # Create a new tensor of the required output shape filled with grey colour
        res = torch.ones(batch_size, 3, new_size, new_size).cuda() * background_color
        # Calculate margin (отступ от края изображения)
        margin = (new_size - new_dimension[1])//2 if tall else (new_size - new_dimension[0])//2
        if tall:
            res[:, :, :, margin:new_size-margin] = resized_tensor
        else:
            res[:, :, margin:new_size - margin, :] = resized_tensor

        return res

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
        for image in images:
            image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            save_path = r"D:\Desktop\system_output\INPUT\1\result"
            cv2.imwrite(os.path.join(save_path, f"{name}.jpg"), image)
            # cv2.imshow("window", image)
            # cv2.waitKey(0)

    @staticmethod
    def show_image(image: np.ndarray) -> None:
        cv2.imshow("", image)
        cv2.waitKey(0)
