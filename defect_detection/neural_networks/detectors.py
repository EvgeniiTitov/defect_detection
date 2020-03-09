from collections import defaultdict
from .detections import DetectedObject, SubImage
import numpy as np
import os


class ObjectDetector(object):
    """
    An idea: One class within which we can detect all objects. Poles, then componenets without having to
    return data back to the CPU twice.

    BLOCKERS:
    1. New weights for poles - 3 classes
    2. New weights for components - 3 classes
    3. Find out how to run inferences without moving data btwn CPU - GPU
    """
    pass


class PolesDetector:
    """
    Class performing utility poles prediction using the YOLOv3 neural net and
    saving objects detected as class objects in a dictionary for subsequent
    usage.
    Image section on which poles have been detected serves the dictionary's key
    role. In this case we consider the whole image.
    As input it accepts a plain image.
    Weights: Pole try 9.
    """
    #path_to_dependencies = r"D:\Desktop\Reserve_NNs\DEPENDENCIES"
    path_to_dependencies = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\dependencies"
    dependencies = "poles"

    def __init__(
            self,
            detector
    ):
        # Initialize predictor - neural network
        self.poles_predictor = detector

        # Detector's dependencies
        config_path = os.path.join(self.path_to_dependencies, self.dependencies + ".cfg")
        weights_path = os.path.join(self.path_to_dependencies, self.dependencies + ".weights")
        classes_path = os.path.join(self.path_to_dependencies, self.dependencies + ".txt")

        # Detector's parameters
        # TO DO: Move it to txt file, so that a user doesn't need to open the code to change it.
        confidence = 0.2
        NMS_thresh = 0.2
        net_res = 416

        # Initialize neural network and prepare it for predictions
        self.poles_predictor.initialize_model(config=config_path,
                                              weights=weights_path,
                                              classes=classes_path,
                                              confidence=confidence,
                                              NMS_threshold=NMS_thresh,
                                              network_resolution=net_res)

        print("Pole detecting network initialized")

    def predict(self, image):
        """
        :param image: Image on which to perform pole detection (the whole original image)
        :return: Dictionary containing all poles detected on the image
        """
        # Create a dictionary to store any poles detected
        poles_detected = defaultdict(list)

        # Specify image section on which predictions take place
        detecting_image_section = SubImage(image, "poles")

        # Call neural net to get predictions. poles - list of lists, each object is represented as
        # a list of 8 items: image index in the batch (0),4BBs coordinates, objectness score,
        # the score of class with max confidence, index on this class
        poles = self.poles_predictor.predict(image)

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

    def modify_box_coordinates(self, image, poles_detected):
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

                poles[0].update_object_coordinates(left=new_left_boundary,
                                                   top=new_top_boundary,
                                                   right=new_right_boundary,
                                                   bottom=new_bot_boundary)
            else:
                for pole in poles:
                    # If we've got 1+ poles on one frame or image, hence the shot was likely taken from
                    # further distance.

                    # TO DO: Overlapping check here. If BBs overlap and a component happens to be in between,
                    # it will be detected twice

                    new_left_boundary = int(pole.BB_left * 0.9)
                    new_right_boundary = int(pole.BB_right * 1.1) if int(pole.BB_right * 1.1) < \
                                                                image.shape[1] else (image.shape[1] - 2)
                    new_top_boundary = int(pole.BB_top * 0.5)
                    new_bot_boundary = int(pole.BB_bottom * 1.1) if int(pole.BB_bottom * 1.1) < \
                                                                image.shape[0] else (image.shape[0] - 2)

                    pole.update_object_coordinates(left=new_left_boundary,
                                                   top=new_top_boundary,
                                                   right=new_right_boundary,
                                                   bottom=new_bot_boundary)


class ComponentsDetector:
    """
    Class performing predictions of utility pole components on image / image
    sections provided.
    All components detected get represented as class objects and are saved in a dictionary
    as values, whereas the image section on which the detection was performed serves the
    role of a dictionary key.
    Weights: Try 6 components
    """
    #path_to_dependencies = r"D:\Desktop\Reserve_NNs\DEPENDENCIES"
    path_to_dependencies = r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\dependencies"
    dependencies_comp = "components"
    dependencies_pil = "pillars"

    def __init__(
            self,
            components_predictor,
            pillar_predictor=None
    ):

        # Initialize components predictor
        self.components_predictor = components_predictor
        # TEMPORARY. Will be replaced with 3 class predictor for concrete poles
        self.pillar_predictor = pillar_predictor

        # Detector's parameters
        # TO DO: Move it to txt file, so that a user doesn't need to open the code to change it.
        confidence = 0.15
        NMS_thresh = 0.25
        net_res = 608

        # Detector's dependencies
        config_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".cfg")
        weights_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".weights")
        classes_path_comp = os.path.join(self.path_to_dependencies, self.dependencies_comp + ".txt")

        # Initialize neural network and prepare it for predictions
        self.components_predictor.initialize_model(config=config_path_comp,
                                                   weights=weights_path_comp,
                                                   classes=classes_path_comp,
                                                   confidence=confidence,
                                                   NMS_threshold=NMS_thresh,
                                                   network_resolution=net_res)

        print("Components detecting network initialized")

        # Pillar detector's dependencies
        config_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".cfg")
        weights_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".weights")
        classes_path_pil = os.path.join(self.path_to_dependencies, self.dependencies_pil + ".txt")

        # Initialize neural network and prepare it for predictions
        self.pillar_predictor.initialize_model(config=config_path_pil,
                                               weights=weights_path_pil,
                                               classes=classes_path_pil,
                                               confidence=confidence,
                                               NMS_threshold=NMS_thresh,
                                               network_resolution=416)

        print("Pillar detecting network initialized")

    def determine_object_class(self, components_detected):
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

    def predict(self, image, pole_predictions):
        """
        Predicts components. Saves them in the appropriate format
        :param image: original image in case no poles have been found
        :param pole_predictions: poles predicted by the pole predicting net (dictionary)
        :return: separate dictionary with components found as values and coordinates of a
        pole on which they were detected as a key.
        """
        # Dictionary to keep components detected
        components_detected = defaultdict(list)

        # If poles detecting neural net detected any poles. Find all components on them
        if pole_predictions:
            # FOR loop below just to play it safe. There should be only one item in the dictionary
            # original image (class object) : poles detected on it (list of lists)
            for window, poles in pole_predictions.items():

                # Consider all poles detected on the original image
                for pole in poles:
                    # For each pole create a separate list for its components to temporary store them
                    components = list()

                    # Crop out the pole detected to send for components detection (modified coordinates)
                    pole_subimage = np.array(window.frame[pole.top:pole.bottom,
                                                          pole.left:pole.right])
                    # Keep track of new subimage cropped out of the original one for components detection
                    pole_image_section = SubImage(frame=pole_subimage,
                                                  name="components")

                    # Save coordinates of this subimage relatively to the original image for
                    # BB drawing and saving objects detected to disk (Relative coordinates)
                    pole_image_section.save_relative_coordinates(top=pole.top,
                                                                 left=pole.left,
                                                                 right=pole.right,
                                                                 bottom=pole.bottom)

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
                                                                DetectedObject(class_id=component[7],
                                                                               confidence=component[5],
                                                                               left=int(component[1]),
                                                                               top=int(component[2]),
                                                                               right=int(component[3]),
                                                                               bottom=int(component[4]))
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
                whole_image = SubImage(image, "components")

                for component in components:
                    components_detected[whole_image].append(
                                                DetectedObject(class_id=component[7],
                                                               confidence=component[5],
                                                               left=int(component[1]),
                                                               top=int(component[2]),
                                                               right=int(component[3]),
                                                               bottom=int(component[4]))
                                                           )

        # TO DO: Could be combined in one function to speed up a bit

        # TO DO: Is the step below even necessary with giving them names?
        if components_detected:
            # Name objects detected by unique names instead of default 0,1,2 etc.
            self.determine_object_class(components_detected)
            # Modify pillar's BB
            self.modify_pillars_BBs(image, components_detected)

        return components_detected


    def modify_pillars_BBs(self, image, componenets_detected):

        for window, components in componenets_detected.items():

            for component in components:
                if component.class_id == 2:

                    # Increase BB's height by moving upper BB border up and bottom down
                    # UP COULD BE FURTHER PULLED
                    # new_top = component.BB_top * 0.8
                    # new_bot = component.BB_bottom * 1.2 if component.BB_bottom * 1.2 <\
                    #                                     image.shape[0] else image.shape[0] - 10

                    new_left = component.BB_left * 0.96
                    new_right = component.BB_right * 1.04 if component.BB_right * 1.04 <\
                                                        image.shape[1] else image.shape[1] - 10

                    # IMPORTANT. Here we overwrite actual object's coordinates that will be used for
                    # drawing a BB around it
                    component.update_object_coordinates(left=int(new_left),
                                                        right=int(new_right))
