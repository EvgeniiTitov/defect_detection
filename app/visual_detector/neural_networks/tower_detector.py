from app.visual_detector.neural_networks.detections_repr import DetectedObject, SubImage
from typing import List, Tuple, Dict
from app.visual_detector.neural_networks.yolo.yolo import YOLOv3
import numpy as np
import os
import torch
import sys


class TowerDetector:
    """
    A wrapper around a neural network to do preprocessing / postprocessing
    Weights: Pole try 9.
    """
    path_to_dependencies = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\app\visual_detector\dependencies"
    dependencies = "poles"
    confidence = 0.2
    NMS_thresh = 0.2
    net_res = 416

    def __init__(self):
        # Network's dependencies
        config_path = os.path.join(self.path_to_dependencies, self.dependencies + ".cfg")
        weights_path = os.path.join(self.path_to_dependencies, self.dependencies + ".weights")
        classes_path = os.path.join(self.path_to_dependencies, self.dependencies + ".txt")
        # Initialize detector - neural network
        self.poles_net = YOLOv3(
            config=config_path,
            weights=weights_path,
            classes=classes_path,
            confidence=TowerDetector.confidence,
            NMS_threshold=TowerDetector.NMS_thresh,
            network_resolution=TowerDetector.net_res
        )
        print("Tower detector successfully initialized")

    def process_batch(self, images_on_gpu: torch.Tensor) -> Dict[SubImage, DetectedObject]:
        """
        Receives a batch of images, runs them through the net to get predictions,
        processes results by representing all detected objects as class objects. Depending
        on class of detected objects, modifies bb.
        :param images:
        :return:
        """
        # To save result for each image in the batch
        batch_detections = {i: {} for i in range(len(images_on_gpu))}

        # Call neural net to get predictions on a batch of images already uploaded to GPU.
        # Returns dict: {image_index_in_batch: [detected poles],...}
        # Each detected pole is represented as a list of 7 items:
        # 4 bbs coordinates, objectness score, confidence, index of this class
        batch_poles_detections = self.poles_net.process_batch(images_on_gpu)

        for i in range(len(batch_poles_detections)):
            # Poles were searched for on original images, not subimages (cropped out objects)
            image_section = SubImage(name="poles")
            # For each image create a key-value pair. Key - image section on which the detection took place - the whole
            # frame in this case. Value - detected poles
            batch_detections[i][image_section] = list()
            # Get nb of towers detected on the image - affects bb modification
            nb_of_poles = len(batch_poles_detections[i])
            # Process predictions for each image in the batch separately
            for pole in batch_poles_detections[i]:
                pole = pole[0]  # yolo returns result as nested lists
                if pole[-1] == 0:
                    class_name = "metal"
                elif pole[-1] == 1:
                    class_name = "concrete"
                elif pole[-1] == 2:
                    class_name = "wood"
                else:
                    print("ERROR: Wrong class index got detected!")
                    continue

                # Represent each detected pole as an object, so that we can easily change its state (adjust
                # BB coordinates) and add more information to it as it moves along the processing pipeline
                pole_detection = DetectedObject(
                    class_id=pole[-1],
                    object_name=class_name,
                    confidence=pole[5],
                    left=int(pole[1]),
                    top=int(pole[2]),
                    right=int(pole[3]),
                    bottom=int(pole[4])
                )
                # Create a copy of pole's bb and modify them (widen)
                self.modify_pole_bb(pole_detection, nb_of_poles, images_on_gpu[i])
                batch_detections[i][image_section].append(pole_detection)

        return batch_detections

    def modify_pole_bb(
            self,
            pole: DetectedObject,
            nb_of_poles: int,
            image: torch.Tensor
    ) -> None:
        """
        Creates a second set of modified pole coordinates - widen bbx
        :param pole:
        :param nb_of_poles:
        :param image:
        :return:
        """
        if nb_of_poles == 1:
            new_left_boundary = int(pole.BB_left * 0.4)
            new_right_boundary = int(pole.BB_right * 1.6) if int(pole.BB_right * 1.6) < \
                                                                 image.shape[1] else (image.shape[1] - 2)
            # Move upper border way up, often when a pole is close up many components do not get
            # included in the box, as a result they do not get found
            new_top_boundary = int(pole.BB_top * 0.1)
            new_bot_boundary = int(pole.BB_bottom * 1.1) if int(pole.BB_bottom * 1.1) < \
                                                                image.shape[0] else (image.shape[0] - 2)
        else:
            new_left_boundary = int(pole.BB_left * 0.9)
            new_right_boundary = int(pole.BB_right * 1.1) if int(pole.BB_right * 1.1) < \
                                                             image.shape[1] else (image.shape[1] - 2)
            new_top_boundary = int(pole.BB_top * 0.5)
            new_bot_boundary = int(pole.BB_bottom * 1.1) if int(pole.BB_bottom * 1.1) < \
                                                            image.shape[0] else (image.shape[0] - 2)

        pole.update_object_coordinates(
            left=new_left_boundary,
            top=new_top_boundary,
            right=new_right_boundary,
            bottom=new_bot_boundary
        )
