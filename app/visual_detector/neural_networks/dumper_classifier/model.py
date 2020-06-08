import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
import torchvision


class DumperClassifier:

    path_to_weights = r"D:\Desktop\system_output\dumper_training\decent\resnet18_Acc1.0_Ftuned1_Pretrained1_OptimizerADAM.pth"
    classes = ["defected", "healthy"]
    img_size = 256, 256

    def __init__(self, load_type: str = "model"):
        # Load entire model
        if load_type == "model":
            try:
                self.model = torch.load(DumperClassifier.path_to_weights)
                self.model.eval()
            except Exception as e:
                print(f"Failed to load the dumpers classifier. Error: {e}")
                raise e

            self.model.cuda()
            print("Dumpers classifier initialized")

        # Load state dict
        else:
            # Use model class and statedict data provided to load the model
            raise NotImplementedError

    def predict(self, images_on_gpu: torch.Tensor) -> list:
        """
        Receives a batch of sliced tensors (not preprocessed in any way) of vibration dumpers and classifies them.
        :param images_on_gpu:
        :return:
        """
        dumpers_to_classify = list()
        # Resize images to the expected size
        for image in images_on_gpu:
            #print("TENSOR BEFORE NORMALIZATION:", image)
            # Normalize image
            try:
                # normalize(
                #     tensor=image,
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],
                #     inplace=True
                # )
                image = self.normalize_image(image)
            except Exception as e:
                print(f"Failed during image normalization for dumper classification. Error: {e}")
                raise e

            #print("TENSOR AFTER NORMALIZATION:", image)

            # Add batch dimension
            img = image.unsqueeze(0)
            try:
                resized_image = F.interpolate(img, size=DumperClassifier.img_size)
            except Exception as e:
                print(f"Failed during dumper tensor resizing. Error: {e}")
                raise e
            dumpers_to_classify.append(resized_image)

        # .cat() them in a batch
        try:
            batch = torch.cat(dumpers_to_classify)
        except Exception as e:
            print(f"Failed during dumper tensors concatination. Error: {e}")
            raise e

        # Get predictions
        try:
            predictions = self.run_forward_pass(batch)
        except Exception as e:
            print(f"Failed while classifying dumpers. Error: {e}")
            raise e

        return predictions

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """

        :param image:
        :return:
        """
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return image_transform(image)

    def run_forward_pass(self, batch: torch.Tensor) -> list:
        """

        :param batch:
        :return:
        """
        with torch.no_grad():
            model_output = self.model(batch)

        labels = [self.classes[out.data.numpy().argmax()] for out in model_output.cpu()]

        return labels
