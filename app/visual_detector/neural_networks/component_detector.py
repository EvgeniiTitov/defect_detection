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
    path_to_dependencies = r"D:\Desktop\branch_dependencies"
    dependencies_comp = "components"
    confidence = 0.15
    NMS_thresh = 0.25
    net_res = 608

    def __init__(self):
        # Initialize components predictor
        config_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".cfg")
        weights_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".weights")
        classes_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".txt")
        # Initialize detector - neural network
        try:
            self.components_net = YOLOv3(
                config=config_path_comp,
                weights=weights_path_comp,
                classes=classes_path_comp,
                confidence=ComponentsDetector.confidence,
                NMS_threshold=ComponentsDetector.NMS_thresh,
                network_resolution=ComponentsDetector.net_res
            )
            print("Components detector successfully initialized")
        except Exception as e:
            print(f"Failed during Component Detector initialization. Error: {e}")
            raise

    def process_batch(self, images_on_gpu: torch.Tensor, towers_predictions: dict) -> dict:
        """
        Receives a batch of images already loaded to GPU and poles detected on them
        Detects components within tower bounding boxes
        :param images_on_gpu:
        :param towers_predictions:
        :return:
        """
        '''
        Input format of detected towers: 
        {
            0: {SubImage1 object (entire frame): [DetectedObject1 (tower), DetectedObject2 (tower)...]}, 
            1: {SubImage2 object (entire frame): [DetectedObject3 (tower)...]}, 
            ...
        }
        For N images in the batch, we get M detected towers. On each tower up to Z components can be detected. 
        1. Slice out all detected towers, bring them all to one size and .cat() them. We need to
           remember how many towers were detected on each image in the batch.
        2. For M towers you get Z component detection. Each detection is to be represented as DetectedObject
        3. Perform matching. Distribute Z detected components among M towers belonging to N images.
        '''
        # Check if any towers have been detected
        detected_towers = list()
        for img_batch_index, detections in towers_predictions.items():
            for subimage, detected_objects in detections.items():
                if detected_objects:
                    detected_towers.extend(obj for obj in detected_objects)
        '''
        1. No towers detected on all images in the batch - for all run entire frames 
        2. On each image in the batch at least one tower detected - crop out, preprocess, predict
        3. Tower(s) detected not on all images in the batch - combine entire frames + towers and predict 
        '''
        print("DETECTED TOWERS:")
        for k, v in towers_predictions.items():
            print(f"Image: {k}. Predictions: {v}")

        sys.exit()

        # If any towers have been detected, search for components within towers bounding boxes
        if detected_towers:
            #TODO: 1. Slice out (crop out) all detected towers from images_on_gpu
            #      2. Remember how many towers found on each image
            #      3. Resize all towers to one YOLO input size
            #      4. .cat() them in one batch
            #      5. Process predictions:
            #         a) Represent objects as DetectedObject, bb as ImageSections
            #         b) Perform matching - what components, belong to what tower, belong to what image in the batch

            detected_components = self.search_components_within_towers()
            raise NotImplementedError("Not yet Eugene")
        # If no towers have been detected, search for components on entire images
        else:
            detected_components = self.search_components_entire_frame(images_on_gpu=images_on_gpu)


    def search_components_entire_frame(self, images_on_gpu: torch.Tensor) -> dict:
        """

        :param images_on_gpu:
        :return:
        """
        batch_detections = {i: {} for i in range(len(images_on_gpu))}
        batch_components_detections = self.components_net.process_batch(images=images_on_gpu)
        print("COMPONENTS DETECTED:", batch_components_detections)

    def search_components_within_towers(self):
        pass


        # # If any pole have been detected
        # if pole_predictions:
        #     # FOR loop below just to play it safe. There should be only one item in the dictionary
        #     # original image (class object) : poles detected on it (list of lists)
        #     for subimage, poles in pole_predictions.items():
        #         # Consider all poles detected on the original image
        #         for pole in poles:
        #             # For each pole create a separate list for its components to temporary store them
        #             components = list()
        #             # Crop out the pole detected to send for components detection (modified coordinates)
        #             pole_subimage = np.array(image[pole.top:pole.bottom, pole.left:pole.right])
        #             # Keep track of new subimage cropped out of the original one for components detection
        #             pole_image_section = SubImage(name="components")
        #             # Save coordinates of this subimage relatively to the original image for
        #             # BB drawing and saving objects detected on disk (Relative coordinates)
        #             pole_image_section.save_relative_coordinates(
        #                 top=pole.top,
        #                 left=pole.left,
        #                 right=pole.right,
        #                 bottom=pole.bottom
        #             )
        #             # ! Depending on the pole's class we want to detect different number of objects
        #             if pole.class_id == 0:  # metal
        #                 predictions = self.components_net.predict(pole_subimage)
        #                 if predictions:
        #                     components += predictions
        #
        #             elif pole.class_id == 1:  # concrete
        #                 # TEMPORARY: Will be replaced with ONE 3 class predictor
        #                 predictions = self.components_net.predict(pole_subimage)
        #                 if predictions:
        #                     components += predictions
        #
        #                 pillar = self.pillar_net.predict(pole_subimage)
        #
        #                 # This since we use 2 nets in sequence predicting 2 and 1 classes, so there is
        #                 # confusion how to keep track what class each object predicted belongs to.
        #                 if pillar:
        #                     assert len(pillar) == 1, "\nERROR: More than 1 pillar detected!"
        #                     # Change class id to 2 for pillars to differ from the rest
        #                     pillar[0][-1] = 2
        #                     components.append(pillar[0])
        #
        #             # Check if any components have been detected on the pole
        #             if components:
        #                 # Represent each component detected as a class object. Save components detected
        #                 # to the dictionary with the appropriate key - image section (pole) on which they
        #                 # were detected
        #                 for component in components:
        #                     components_detected[pole_image_section].append(
        #                                                         DetectedObject(
        #                                                             class_id=component[7],
        #                                                             confidence=component[5],
        #                                                             left=int(component[1]),
        #                                                             top=int(component[2]),
        #                                                             right=int(component[3]),
        #                                                             bottom=int(component[4])
        #                                                         )
        #                     )
        #
        # else:
        #     # In case no poles have been detected, send the whole image for components detection
        #     # in case there are any close-up components on the image
        #     components = list()
        #     predictions = self.components_net.predict(image)
        #
        #     if predictions:
        #         components += predictions
        #
        #     # TEMPORARY:
        #     pillar = self.pillar_net.predict(image)
        #
        #     #assert pillar is None or len(pillar) == 1, "ERROR: More than 1 pillar detected"
        #     if pillar and len(pillar) > 1:
        #         print("ATTENTION: MORE THAN ONE PILLAR DETECTED!")
        #
        #     # Change its class to 2 (separate net, will be changed after retraining)
        #     if pillar:
        #         pillar[0][-1] = 2
        #         components.append(pillar[0])
        #
        #     if components:
        #         whole_image = SubImage("components")
        #
        #         for component in components:
        #             components_detected[whole_image].append(
        #                                         DetectedObject(
        #                                             class_id=component[7],
        #                                             confidence=component[5],
        #                                             left=int(component[1]),
        #                                             top=int(component[2]),
        #                                             right=int(component[3]),
        #                                             bottom=int(component[4])
        #                                         )
        #             )
        #
        # if components_detected:
        #     # Name objects detected by unique names instead of default 0,1,2 etc.
        #     self.determine_object_class(components_detected)
        #     # Modify pillar's BB
        #     self.modify_pillars_BBs(image, components_detected)
        #
        # return components_detected

    def modify_pillars_BBs(self, image: np.ndarray, componenets_detected: dict) -> None:
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

        return
