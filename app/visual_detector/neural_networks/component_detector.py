from collections import defaultdict
from app.visual_detector.neural_networks.detections_repr import DetectedObject, SubImage
from typing import List, Dict, Tuple
from app.visual_detector.neural_networks.yolo.yolo import YOLOv3
import numpy as np
import os
import torch
import sys


class ComponentsDetector:
    """
    A wrapper around a neural network to do preprocessing / postprocessing
    Weights: Try 6 components
    """
    path_to_dependencies = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\app\visual_detector\dependencies"
    dependencies_comp = "components"
    dependencies_pil = "pillars"
    confidence = 0.15
    NMS_thresh = 0.25
    net_res = 608

    def __init__(self):
        # Initialize components predictor
        config_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".cfg")
        weights_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".weights")
        classes_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".txt")
        # Initialize detector - neural network
        self.components_net = YOLOv3(
            config=config_path_comp,
            weights=weights_path_comp,
            classes=classes_path_comp,
            confidence=ComponentsDetector.confidence,
            NMS_threshold=ComponentsDetector.NMS_thresh,
            network_resolution=ComponentsDetector.net_res
        )
        print("Components detector successfully initialized")

        # TEMPORARY. Will be replaced with 3 class predictor for concrete poles
        # Initialize pole predictor
        config_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".cfg")
        weights_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".weights")
        classes_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".txt")
        self.pillar_net = YOLOv3(
            config=config_path_pil,
            weights=weights_path_pil,
            classes=classes_path_pil,
            confidence=ComponentsDetector.confidence,
            NMS_threshold=ComponentsDetector.NMS_thresh,
            network_resolution=416
        )
        print("Pillar detector successfully initialized")

    def determine_object_class(self, components_detected: dict) -> None:
        """
        Checks object's class and names it. Since we've got multiple nets predicting objects
        like 0,1,2 classes, we want to make sure it doesn't get confusing during saving data and
        drawing BBs
        :return: Nothing. Changes object's state
        """
        for subimage, components in components_detected.items():

            for component in components:
                if component.class_id == 0:
                    component.object_name = "insl"  # Insulator

                elif component.class_id == 1:
                    component.object_name = "dump"  # Vibration dumper

                else:
                    component.object_name = "pillar"

    def process_batch(self, image_on_gpu: torch.Tensor, pole_predictions: dict) -> dict:
        """
        Receives a batch of images already loaded to GPU and poles detected on them
        Detects components within tower bounding boxes
        :param image_on_gpu:
        :param pole_predictions:
        :return:
        """
        # Dictionary to keep components detected
        components_detected = defaultdict(list)

        # If any pole have been detected
        if pole_predictions:
            # FOR loop below just to play it safe. There should be only one item in the dictionary
            # original image (class object) : poles detected on it (list of lists)
            for subimage, poles in pole_predictions.items():
                # Consider all poles detected on the original image
                for pole in poles:
                    # For each pole create a separate list for its components to temporary store them
                    components = list()
                    # Crop out the pole detected to send for components detection (modified coordinates)
                    pole_subimage = np.array(image[pole.top:pole.bottom, pole.left:pole.right])
                    # Keep track of new subimage cropped out of the original one for components detection
                    pole_image_section = SubImage(name="components")
                    # Save coordinates of this subimage relatively to the original image for
                    # BB drawing and saving objects detected on disk (Relative coordinates)
                    pole_image_section.save_relative_coordinates(
                        top=pole.top,
                        left=pole.left,
                        right=pole.right,
                        bottom=pole.bottom
                    )
                    # ! Depending on the pole's class we want to detect different number of objects
                    if pole.class_id == 0:  # metal
                        predictions = self.components_net.predict(pole_subimage)
                        if predictions:
                            components += predictions

                    elif pole.class_id == 1:  # concrete
                        # TEMPORARY: Will be replaced with ONE 3 class predictor
                        predictions = self.components_net.predict(pole_subimage)
                        if predictions:
                            components += predictions

                        pillar = self.pillar_net.predict(pole_subimage)

                        # This since we use 2 nets in sequence predicting 2 and 1 classes, so there is
                        # confusion how to keep track what class each object predicted belongs to.
                        if pillar:
                            assert len(pillar) == 1, "\nERROR: More than 1 pillar detected!"
                            # Change class id to 2 for pillars to differ from the rest
                            pillar[0][-1] = 2
                            components.append(pillar[0])

                    # Check if any components have been detected on the pole
                    if components:
                        # Represent each component detected as a class object. Save components detected
                        # to the dictionary with the appropriate key - image section (pole) on which they
                        # were detected
                        for component in components:
                            components_detected[pole_image_section].append(
                                                                DetectedObject(
                                                                    class_id=component[7],
                                                                    confidence=component[5],
                                                                    left=int(component[1]),
                                                                    top=int(component[2]),
                                                                    right=int(component[3]),
                                                                    bottom=int(component[4])
                                                                )
                            )

        else:
            # In case no poles have been detected, send the whole image for components detection
            # in case there are any close-up components on the image
            components = list()
            predictions = self.components_net.predict(image)

            if predictions:
                components += predictions

            # TEMPORARY:
            pillar = self.pillar_net.predict(image)

            #assert pillar is None or len(pillar) == 1, "ERROR: More than 1 pillar detected"
            if pillar and len(pillar) > 1:
                print("ATTENTION: MORE THAN ONE PILLAR DETECTED!")

            # Change its class to 2 (separate net, will be changed after retraining)
            if pillar:
                pillar[0][-1] = 2
                components.append(pillar[0])

            if components:
                whole_image = SubImage("components")

                for component in components:
                    components_detected[whole_image].append(
                                                DetectedObject(
                                                    class_id=component[7],
                                                    confidence=component[5],
                                                    left=int(component[1]),
                                                    top=int(component[2]),
                                                    right=int(component[3]),
                                                    bottom=int(component[4])
                                                )
                    )

        if components_detected:
            # Name objects detected by unique names instead of default 0,1,2 etc.
            self.determine_object_class(components_detected)
            # Modify pillar's BB
            self.modify_pillars_BBs(image, components_detected)

        return components_detected

    def modify_pillars_BBs(
            self,
            image: np.ndarray,
            componenets_detected: dict
    ) -> None:
        """
        Slightly widens pillar's bb in order to ensure both edges are within the box
        :param image:
        :param componenets_detected:
        :return:
        """
        for window, components in componenets_detected.items():
            for component in components:
                if component.class_id == 2:
                    new_left = component.BB_left * 0.96
                    new_right = component.BB_right * 1.04 if component.BB_right * 1.04 <\
                                                        image.shape[1] else image.shape[1] - 10
                    component.update_object_coordinates(left=int(new_left), right=int(new_right))
