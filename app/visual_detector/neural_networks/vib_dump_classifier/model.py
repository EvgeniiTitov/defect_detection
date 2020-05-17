import torch
from torch.nn import Module
from torchvision import transforms
import torchvision
import sys
import cv2
from typing import List, Dict
import torch.nn.functional as F
from PIL import Image


class DumperClassifier:
    """

    """
    def __init__(
            self,
            load_type: str,
            path_to_data: str,
            classes: list,
            model_class=None
    ):
        # Load entire model
        if load_type == "model":
            print("Attempting to load the dumper classifier...")
            try:
                self.model = torch.load(path_to_data)
                self.model.eval()
                self.model.cuda()
            except Exception as e:
                print(f"Failed to load the dumpers classifier. Error: {e}")
                raise
            print("Dumpers classifier initialized")

        # Load state dict
        else:
            # Use model class and statedict data provided to load the model
            raise NotImplementedError
        self.classes = classes

    def predict_batch(self, batch_of_images: List[Image.Image]) -> list:
        """
        Receives a list of images (cropped out dumpers) and classifies them
        """
        image_tensors = list()
        # preprocess images and construct a batch
        for image in batch_of_images:
            image_tensor = DumperClassifier.preproccess_image(image)
            image_tensor.unsqueeze_(0)
            image_tensors.append(image_tensor)

        images_batch = torch.cat(image_tensors)
        try:
            images_batch = images_batch.cuda()
        except Exception as e:
            print(f"Failed to move batch to GPU. Error: {e}")
            raise

        labels = self.run_forward_pass(images_batch)

        return labels

    @staticmethod
    def preproccess_image(image: Image.Image) -> torch.Tensor:
        """
        Receives an image opened with PIL.Image. Resizes, converts to torch.Tensor and normalizes
        :param image:
        :return:
        """
        image_transforms = torchvision.transforms.Compose([
                    transforms.Resize((255, 255)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )]
        )
        return image_transforms(image)

    def _coord_preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Receives an image of the torch.Tensor format. Resizes and normalizes it.
        :param image:
        :return:
        """
        image_transforms = torchvision.transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )]
        )
        return image_transforms(image)

    def predict_using_coord(
            self,
            images_on_gpu: torch.Tensor,
            coordinates: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Receives images already uploaded in GPU alongside the dictionary of coordinates. Coordinates
        represent parts of the images that need to be sliced out and run through the network for
        classification
        :param images_on_gpu:
        :param coordinates:
        :return:
        """
        assert len(images_on_gpu) == len(coordinates), "Nb of images != sets of coordinates"
        keys = list(coordinates.keys())
        output_format = {
            "nb_of_dumpers": 0,
            "status": []
        }
        output = {key: output_format for key in keys}

        # Define regions on the images using coordinates provided that need to be run through
        # the network to get classified: defected / not-defected vibration dumper
        subimages_to_process = list()
        for i in range(len(images_on_gpu)):
            image = images_on_gpu[i]
            coord = coordinates[keys[i]]
            # Loop over provided coordinates, and collect all subimages using the
            # provided coordinates
            # Keep track of how many subimages created on each image
            counter = 0
            for value in coord.values():
                # Slice (crop out) new subimage containing an object to classify
                top = value[0]
                bottom = value[1]
                left = value[2]
                right = value[3]
                subimage = image[:, top:bottom, left:right]
                # Resize an image and append to images to process
                subimage = subimage.unsqueeze(dim=0)  # interpolate requires extra dim
                resized_subimage = F.interpolate(subimage, size=(256, 256))
                subimages_to_process.append(resized_subimage)
                counter += 1

            output[keys[i]]["nb_of_dumpers"] = counter
        # BAG HERE - doesnt pop all results - matching issue because nb of dumlers always 2

        print(output)

        # Create a batch
        try:
            batch = torch.cat(subimages_to_process)
        except Exception as e:
            print(f"Failed during batch concatination. Error: {e}")
            raise

        # Visualize slices subimages
        #TrainedModel.visualise_sliced_img(subimages_to_process)

        # Run NN, get predictions
        predictions = self.run_forward_pass(batch)

        # Match predictions with appropriate images
        for key, value in output.items():
            nb_of_dumpers = value["nb_of_dumpers"]
            items = list()
            for i in range(nb_of_dumpers):
                items.append(predictions.pop(0))
            value["status"] = items
            assert nb_of_dumpers == len(value["status"]), "ERROR: Matching went wrong"
        assert not predictions, "ERROR: Not all results have been matched"

        return output

    def run_forward_pass(self, images_on_gpu: torch.Tensor) -> list:
        """

        :param images_on_gpu:
        :return:
        """
        with torch.no_grad():
            model_output = self.model(images_on_gpu)

        labels = [self.classes[out.data.numpy().argmax()] for out in model_output.cpu()]

        return labels

    @staticmethod
    def visualise_sliced_img(images: List[torch.Tensor]) -> None:
        """
        Visualise sliced (cropped out) images.
        :param images:
        :return:
        """
        for image in images:
            image = image.squeeze()
            image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            cv2.imshow("window", image)
            cv2.waitKey(0)
