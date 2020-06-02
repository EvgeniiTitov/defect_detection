from app.visual_detector.neural_networks.detections_repr import DetectedObject, SubImage
from typing import Dict
from app.visual_detector.neural_networks.yolo.yolo import YOLOv3
from app.visual_detector.utils import TensorManager
import os
import torch


class TowerDetector:
    """
    A wrapper around a neural network to do preprocessing / postprocessing of data before
    and after the neural network to detect towers
    Weights: Pole try 9.
    """
    path_to_dependencies = r"D:\Desktop\branch_dependencies"
    dependencies = "poles"
    confidence = 0.2
    NMS_thresh = 0.2
    net_res = 512

    def __init__(self):
        # Network's dependencies
        config_path = os.path.join(self.path_to_dependencies, self.dependencies + ".cfg")
        weights_path = os.path.join(self.path_to_dependencies, self.dependencies + ".weights")
        classes_path = os.path.join(self.path_to_dependencies, self.dependencies + ".txt")

        # Initialize detector - neural network
        try:
            self.poles_net = YOLOv3(
                config=config_path,
                weights=weights_path,
                classes=classes_path,
                confidence=TowerDetector.confidence,
                NMS_threshold=TowerDetector.NMS_thresh,
                network_resolution=TowerDetector.net_res
            )
            print("Tower detector successfully initialized")
        except Exception as e:
            print(f"Failed during Tower Detector initialization. Error: {e}")
            raise e

    def process_batch(self, images_on_gpu: torch.Tensor) -> Dict[SubImage, DetectedObject]:
        """
        Receives a batch of images, runs them through the net to get predictions,
        processes results by representing all detected objects as class objects. Depending
        on class of detected objects, modifies bb.
        :param images_on_gpu:
        :return:
        """
        # Preprocess images
        preprocessed_imgs, original_shape = TensorManager.resize_tensor_keeping_aspratio(
            batch_tensor=images_on_gpu,
            new_size=TowerDetector.net_res
        )
        preprocessed_imgs.div_(255.0)

        # Get tower detections
        tower_detections = self.poles_net.process_batch(preprocessed_imgs)

        # Rescace bounding boxes relatively to images of the original dimensions
        recalculated_detections = TensorManager.rescale_bb(
            detections=tower_detections,
            current_dim=TowerDetector.net_res,
            original_shape=original_shape
        )
        # Postprocess results - represent all detections as class objects for convenience
        detections_output = {i: {} for i in range(len(images_on_gpu))}
        for i in range(len(recalculated_detections)):
            # Poles were searched for on original images, not subimages (cropped out objects)
            image_section = SubImage(name="frame")
            # For each image create a key-value pair. Key - image section on which the detection took place - the whole
            # frame in this case. Value - detected poles
            detections_output[i][image_section] = list()
            # Process predictions for each image in the batch separately
            for pole in recalculated_detections[i]:
                if pole[-1] == 0:
                    class_name = "concrete"
                elif pole[-1] == 1:
                    class_name = "metal"
                elif pole[-1] == 2:
                    class_name = "wood"
                else:
                    print("ERROR: Wrong class index got detected!")
                    continue
                # Represent each detected pole as an object, so that we can easily change its state (adjust
                # BB coordinates) and add more information to it as it moves along the processing pipeline
                try:
                    pole_detection = DetectedObject(
                        left=int(pole[0]),
                        top=int(pole[1]),
                        right=int(pole[2]),
                        bottom=int(pole[3]),
                        class_id=pole[-1],
                        object_name=class_name,
                        confidence=pole[5]
                    )
                except Exception as e:
                    print(f"Failed during DetectedObject initialization. Error: {e}")
                    raise e
                detections_output[i][image_section].append(pole_detection)
        '''
        Output format if any towers detected: 
        {
            0: {SubImage object: [DetectedObject object, DetectedObject object...]}, 
            1: {SubImage object: [DetectedObject object...]}, 
            ...
        }
        '''
        return detections_output
