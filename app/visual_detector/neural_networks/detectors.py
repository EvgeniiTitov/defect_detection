from collections import defaultdict
from app.visual_detector.neural_networks.detections_repr import DetectedObject, SubImage
from typing import List
from app.visual_detector.neural_networks.yolo.yolo import YOLOv3
import numpy as np
import os
import torch


class PolesDetector:
    """
    Class performing poles detection using the YOLOv3 neural net. A wrapper around the actual
    network that can do preprocessing / postprocessing of images / predictions.
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
        # Initialize predictor - neural network
        self.poles_predictor = YOLOv3(
            config=config_path,
            weights=weights_path,
            classes=classes_path,
            confidence=PolesDetector.confidence,
            NMS_threshold=PolesDetector.NMS_thresh,
            network_resolution=PolesDetector.net_res
        )
        print("Pole detector initialized")

    # def predict(self, image):
    #     " DELETE ME I AM FOR DEBUGGING"
    #     poles = self.poles_predictor.predict(image)
    #     print("PREDICTIONS:", poles)

    def predict_batch(self, images: List[np.ndarray]) -> dict:
        """
        :param images: Batch of images concatinated and moved to GPU
        :return:
        """
        # Create a dictionary to store any poles detected
        poles_detected = defaultdict(list)
        # Specify image section on which predictions take place
        detecting_image_section = SubImage(name="poles")


        # Call neural net to get predictions. poles - list of lists, each object is represented as
        # a list of 8 items: image index in the batch (0),4BBs coordinates, objectness score,
        # the score of class with max confidence, index on this class
        poles, batch_on_gpu = self.poles_predictor.predict_batch(images)
        print("Predicted poles for a batch:")
        for pole in poles:
            print(pole)

        import sys
        sys.exit()

        # Represent each object detected as a class object. Add all objects
        # to the dictionary as values.
        # Check if any poles have been detected otherwise return the empty dictionary
        if poles:
            for pole in poles:

                # Check what class it is. Do it this way, otherwise after 2 nets in sequence you will have
                # objects with class ids 0,1 and 0,1,2. Objects from both nets mix together. Add extra attr
                if pole[7] == 0:
                    class_name = "metal"
                else:
                    class_name = "concrete"

                # Save poles detected as class objects - DetectedObject
                poles_detected[detecting_image_section].append(DetectedObject(class_id=pole[7],
                                                                              object_name=class_name,
                                                                              confidence=pole[5],
                                                                              left=int(pole[1]),
                                                                              top=int(pole[2]),
                                                                              right=int(pole[3]),
                                                                              bottom=int(pole[4])))

            # Modify poles coordinates to widen them for better components detection (some might stick out, so
            # they don't end up inside the objects BB. Hence, they get missed since component detection happens only
            # inside the objects box.
            self.modify_box_coordinates(image, poles_detected)

        return poles_detected

    def modify_box_coordinates(
            self,
            image: np.ndarray,
            poles_detected: dict
    ) -> None:
        """
        Modifies pole's BB. 50% both sides if only one pole detected (likely to be closeup), 10% if more
        :param image: image on which detection of poles took place (original image)
        :param poles_detected: detections of poles
        Will be used to make sure new modified coordinates do not go beyond image's edges
        :return: None. Simply modified coordinates
        """
        for window, poles in poles_detected.items():
            # Let's consider all poles detected on an image and modify their coordinates.
            # If only one pole's been detected, just widen the box 60% both sides
            if len(poles) == 1:
                new_left_boundary = int(poles[0].BB_left * 0.4)
                new_right_boundary = int(poles[0].BB_right * 1.6) if int(poles[0].BB_right * 1.6) <\
                                                                    image.shape[1] else (image.shape[1] - 2)
                # Move upper border way up, often when a pole is close up many components do not get
                # included in the box, as a result they do not get found
                new_top_boundary = int(poles[0].BB_top * 0.1)
                new_bot_boundary = int(poles[0].BB_bottom * 1.1) if int(poles[0].BB_bottom * 1.1) <\
                                                                    image.shape[0] else (image.shape[0] - 2)

                poles[0].update_object_coordinates(
                    left=new_left_boundary,
                    top=new_top_boundary,
                    right=new_right_boundary,
                    bottom=new_bot_boundary
                )

            else:
                for pole in poles:
                    # If we've got 1+ poles on one frame or image, hence the shot was likely taken from
                    # further distance.

                    # TODO: Overlapping check here. If BBs overlap and a component happens to be in between,
                    # it will be detected twice

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


class ComponentsDetector:
    """
    Class performing predictions of utility pole components on image / image sections provided.
    All components detected get represented as class objects and are saved in a dictionary
    as values, whereas the image section on which the detection was performed serves the
    role of a dictionary key.
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
        self.components_predictor = YOLOv3(
            config=config_path_comp,
            weights=weights_path_comp,
            classes=classes_path_comp,
            confidence=ComponentsDetector.confidence,
            NMS_threshold=ComponentsDetector.NMS_thresh,
            network_resolution=ComponentsDetector.net_res
        )
        print("Components detector initialized")

        # TEMPORARY. Will be replaced with 3 class predictor for concrete poles
        # Initialize pole predictor
        config_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".cfg")
        weights_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".weights")
        classes_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".txt")
        self.pillar_predictor = YOLOv3(
            config=config_path_pil,
            weights=weights_path_pil,
            classes=classes_path_pil,
            confidence=ComponentsDetector.confidence,
            NMS_threshold=ComponentsDetector.NMS_thresh,
            network_resolution=416
        )
        print("Pillar detector initialized")

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

    def predict(
            self,
            image: np.ndarray,
            pole_predictions: dict
    ) -> dict:
        """
        Predicts components. Saves them in the appropriate format
        :param image: original image in case no poles have been found
        :param pole_predictions: poles predicted by the pole predicting net (dictionary)
        :return: separate dictionary with components found as values and coordinates of a
        pole on which they were detected as a key.
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
                        predictions = self.components_predictor.predict(pole_subimage)
                        if predictions:
                            components += predictions

                    elif pole.class_id == 1:  # concrete
                        # TEMPORARY: Will be replaced with ONE 3 class predictor
                        predictions = self.components_predictor.predict(pole_subimage)
                        if predictions:
                            components += predictions

                        pillar = self.pillar_predictor.predict(pole_subimage)

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
            predictions = self.components_predictor.predict(image)

            if predictions:
                components += predictions

            # TEMPORARY:
            pillar = self.pillar_predictor.predict(image)

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
