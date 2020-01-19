from collections import defaultdict
from .detections import DetectedObject, DetectionImageSection
import numpy as np


class ComponentsDetector:
    """
    Class performing predictions of utility pole components on image / image
    sections provided.
    All components detected get represented as class objects and are saved in a dictionary
    as values, whereas the image section on which the detection was performed serves the
    role of a dictionary key.
    """
    def __init__(
            self,
            components_predictor,
            pillar_predictor=None
    ):
        # Initialize components predictor
        self.components_predictor = components_predictor
        # TEMPORARY. Will be replaced with 3 class predictor for concrete poles
        self.pillar_predictor = pillar_predictor

    def determine_object_class(self, components_detected):
        """
        Checks object's class and names it. Since we've got multiple nets predicting objects
        like 0,1,2 classes, we want to make sure we don't get confused later.
        Dynamically creates new attribute
        :return: Nothing. Simply names objects
        """
        for window, components in components_detected.items():

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
                    pole_image_section = DetectionImageSection(pole_subimage, "components")

                    # Save coordinates of this subimage relatively to the original image for
                    # painless BB drawing and saving objects detected to disk (Relative coordinates)
                    pole_image_section.save_relative_coordinates(pole.top,
                                                                 pole.left,
                                                                 pole.right,
                                                                 pole.bottom)

                    # ! Depending on the pole's class we want to detect different number of objects
                    if pole.class_id == 0:  # metal
                        components += self.components_predictor.predict(pole_subimage)

                    elif pole.class_id == 1:  # concrete
                        # TEMPORARY: Will be replaced with ONE 3 class predictor
                        components += self.components_predictor.predict(pole_subimage)

                        pillar = self.pillar_predictor.predict(pole_subimage)

                        # REMOVE ME AFTER TESTS AND RESEARCH
                        if len(pillar) > 1:
                            print("WARNING: MORE THAN ONE PILLAR DETECTED")

                        # This since we use 2 nets in sequence predicting 2 and 1 classes, so there is
                        # confusion how to keep track what class each object predicted belongs to.
                        if pillar:
                            components.append([2, pillar[0][1], pillar[0][2],
                                               pillar[0][3], pillar[0][4], pillar[0][5]])

                    # Check if any components have been detected on the pole
                    if components:
                        # Represent each component detected as a class object. Save components detected
                        # to the dictionary with the appropriate key - image section (pole) on which they
                        # were detected
                        for component in components:
                            components_detected[pole_image_section].append(
                                DetectedObject(component[0], component[1], component[2],
                                               component[3], component[4], component[5])
                                                                          )

        else:
            # In case no poles have been detected, send the whole image for components detection
            # in case there are any close-up components on the image
            components = self.components_predictor.predict(image)

            # TEMPORARY:
            pillar = self.pillar_predictor.predict(image)

            if len(pillar) > 1:
                print("WARNING: MORE THAN ONE PILLAR DETECTED")

            if pillar:
                components.append([2, pillar[0][1], pillar[0][2],
                                   pillar[0][3], pillar[0][4], pillar[0][5]])

            if components:
                whole_image = DetectionImageSection(image, "components")

                for component in components:
                    components_detected[whole_image].append(
                        DetectedObject(component[0], component[1], component[2],
                                       component[3], component[4], component[5])
                                                            )

        # TO DO: Could be combined in one function to speed up a bit
        # Name objects detected by unique names instead of default 0,1,2 etc.
        self.determine_object_class(components_detected)
        # Modify pillar's BB
        # self.modify_pillars_BBs(components_detected)

        return components_detected


    def modify_pillars_BBs(self, componenets_detected):

        for window, components in componenets_detected.items():

            for component in components:
                if component.class_id == 2:
                    pass
                    # CHECK IF IT IS NECESSARY AT ALL


class PoleDetector:
    """
    Class performing utility poles prediction using the YOLOv3 neural net and
    saving objects detected as class objects in a dictionary for subsequent
    usage.
    Image section on which poles have been detected serves the dictionary's key
    role. In this case we consider the whole image.
    As input it accepts a plain image.
    """
    def __init__(self, predictor):
        # Initialize predictor
        self.poles_predictor = predictor

    def determine_object_class(self, poles_detected):
        """
        Checks object's class and names it. Since we've got multiple nets predicting objects
        like 0,1,2 classes, we want to make sure we don't get confused later.
        Dynamically creates new attribute
        :return: Nothing. Simply names objects
        """
        for window, poles in poles_detected.items():
            for pole in poles:
                if pole.class_id == 0:
                    pole.object_name = "metal"
                else:
                    pole.object_name = "concrete"

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
            # If only one pole's been detected, just widen the box 50% both sides
            if len(poles) == 1:
                new_left_boundary = int(poles[0].BB_left * 0.5)
                new_right_boundary = int(poles[0].BB_right * 1.5) if int(poles[0].BB_right * 1.5) <\
                                                                    image.shape[1] else (image.shape[1] - 2)
                new_top_boundary = int(poles[0].BB_top * 0.9)
                new_bot_boundary = int(poles[0].BB_bottom * 1.1) if int(poles[0].BB_bottom * 1.1) <\
                                                                    image.shape[0] else (image.shape[0] - 2)

                poles[0].update_object_coordinates(left=new_left_boundary,
                                                   top=new_top_boundary,
                                                   right=new_right_boundary,
                                                   bottom=new_bot_boundary)
            else:
                for pole in poles:

                    # ! CHECK FOR OVERLAPPING

                    new_left_boundary = int(pole.BB_left * 0.9)
                    new_right_boundary = int(pole.BB_right * 1.1) if int(pole.BB_right * 1.1) < \
                                                                image.shape[1] else (image.shape[1] - 2)
                    new_top_boundary = int(pole.BB_top * 0.9)
                    new_bot_boundary = int(pole.BB_bottom * 1.1) if int(pole.BB_bottom * 1.1) < \
                                                                image.shape[0] else (image.shape[0] - 2)

                    pole.update_object_coordinates(left=new_left_boundary,
                                                   top=new_top_boundary,
                                                   right=new_right_boundary,
                                                   bottom=new_bot_boundary)

    def predict(self, image):
        """
        :param image: Image on which to perform pole detection (the whole original image)
        :return: Dictionary containing all poles detected on the image
        """
        # Create a dictionary to keep the predictions made
        poles_detected = defaultdict(list)

        # Specify image section on which predictions take place
        detecting_image_section = DetectionImageSection(image, "poles")

        # Call neural net to get predictions. poles - list of lists, each object is represented as
        # a list of 6 items: class, confidence, coordinates
        poles = self.poles_predictor.predict(image)

        # Represent each object detected as a class object. Add all objects
        # to the dictionary as values.
        # Check if any poles have been detected otherwise return the empty dictionary
        if poles:
            for pole in poles:
                poles_detected[detecting_image_section].append(
                        DetectedObject(pole[0], pole[1], pole[2], pole[3], pole[4], pole[5])
                                                               )
            # Modify poles coordinates to widen them for better components detection (some might stick out, so
            # they don't end up inside the objects BB. Hence, they get missed since component detection happens only
            # inside the objects box.
            self.modify_box_coordinates(image, poles_detected)
            # Name objects detected by unique names instead of default 0,1,2 etc.
            self.determine_object_class(poles_detected)

        return poles_detected
