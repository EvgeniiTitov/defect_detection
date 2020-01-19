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
    def __init__(self, predictor):
        # Initialize components predictor
        self.components_predictor = predictor

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
                else:
                    component.object_name = "dump"  # Vibration dumper

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
                    components = list()
                    # Crop out the pole detected to send for components detection (modified coordinates)
                    pole_subimage = np.array(window.frame[pole.top:pole.bottom,
                                                          pole.left:pole.right])
                    pole_image_section = DetectionImageSection(pole_subimage, "components")
                    # Save coordinates of this subimage relatively to the original image for
                    # painless BB drawing and saving objects detected to disk
                    pole_image_section.save_relative_coordinates(pole.top, pole.left,
                                                                 pole.right, pole.bottom)

                    # ! Depending on the pole's class we want to detect different number of objects
                    # ! FOR NOW IT IS THE SAME NN SINCE WE DO NOT HAVE WEIGHTS YET
                    if pole.class_id == 0:  # metal
                        components += self.components_predictor.predict(pole_subimage)
                    elif pole.class_id == 1:  # concrete
                        components += self.components_predictor.predict(pole_subimage)
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
                        self.determine_object_class(components_detected)
        else:
            # In case no poles have been detected, send the whole image for components detection
            # in case there are any close-up components on the image
            components = self.components_predictor.predict(image)
            if components:
                whole_image = DetectionImageSection(image, "components")
                for component in components:
                    components_detected[whole_image].append(
                        DetectedObject(component[0], component[1], component[2],
                                       component[3], component[4], component[5])
                                                            )
                #  Name objects detected by unique names instead of default 0,1,2 etc.
                self.determine_object_class(components_detected)

        return components_detected


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

        # Call neural net to get predictions
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


class PillarDetector:
    """
    Class performing pillar detection
    """
    def __init__(self, predictor):
        # Initialize
        self.pillar_predictor = predictor

    def predict(self, image, poles_detected):
        """
        Method running pillars predictions.
        :param image: Image to crop out image sections containing the poles using the coordinates
        of the poles predicted
        :return:
        """
        pillars_detected = defaultdict(list)

        if poles_detected:
            # FOR loop just to play it safe. We will have only one image section for now - the whole image on
            # which we attempted to detect pole(s)
            for window, poles in poles_detected.items():
                # One an image we might have multiple concrete poles. For each we can potentially
                # detect a pillar
                for pole in poles:
                    # Skip metal poles.
                    if pole.object_name == "metal":
                        continue
                    # Crop out image section containing a pole predicted (using its coordinates from YOLO)
                    # ! Here we don't really need to use the widened left and right coordinates since the
                    # pole pillar can only be inside the pole's box unlike dumpers and insulators that stick out
                    pole_subimage = np.array(window.frame[pole.top:pole.bottom,
                                                          pole.left:pole.right])
                    pole_image_section = DetectionImageSection(pole_subimage, "pillars")
                    # Save coordinates of the subimage relatively to the original image
                    pole_image_section.save_relative_coordinates(pole.top, pole.left, pole.right, pole.bottom)
                    # Detect pillars in this subimage containing a concrete pole
                    # Pillar is just a list of lists (there needs to be just one predicted!)
                    pillar = self.pillar_predictor.predict(pole_subimage)
                    # There's supposed to be only one pillar for each concrete pole,
                    # process the results that are simple list of lists outputted by YOLO
                    if pillar:
                        # Check if more than one pillar got predicted for one pole
                        if len(pillar) > 1:
                            print("WARNING: More than 1 pillar got detected")
                            # Find the one with the highest confidence and select it.
                            the_pillar = (0, 0)  # index, confidence
                            for index, plr in enumerate(pillar):
                                if plr[1] > the_pillar[-1]:
                                    the_pillar = (index, plr[1])

                            # Once we've found the one. Save it
                            index_best = the_pillar[0]
                            pillars_detected[pole_image_section].append(
                                    DetectedObject(pillar[index_best][0], pillar[index_best][1],
                                                   pillar[index_best][2], pillar[index_best][3],
                                                   pillar[index_best][4], pillar[index_best][5])
                                                                        )

                        else:
                            # One object, still use for loop for convenience
                            pillars_detected[pole_image_section].append(
                                    DetectedObject(pillar[0][0], pillar[0][1], pillar[0][2],
                                                   pillar[0][3], pillar[0][4], pillar[0][5])
                                                                        )
        else:
            # Make pillar detection on the whole image (since a concrete pole has not been detected)
            pillar = self.pillar_predictor.predict(image)
            # pillar - list of lists
            if pillar:
                image_section = DetectionImageSection(image, "pillars")
                if len(pillar) > 1:
                    print("WARNING: More than one pillar got detected")
                    the_pillar = (0, 0)  # index, confidence
                    for index, plr in enumerate(pillar):
                        if plr[1] > the_pillar[-1]:
                            the_pillar = (index, plr[1])
                    index_best = the_pillar[0]
                    pillars_detected[image_section].append(
                        DetectedObject(pillar[index_best][0], pillar[index_best][1],
                                       pillar[index_best][2], pillar[index_best][3],
                                       pillar[index_best][4], pillar[index_best][5])
                                                           )
                else:
                    pillars_detected[image_section].append(
                        DetectedObject(pillar[0][0], pillar[0][1], pillar[0][2],
                                       pillar[0][3], pillar[0][4], pillar[0][5])
                                                           )

        # Postprocess the pillars predicted (name them, cut the BBs)
        if pillars_detected:
            self.determine_object_class(pillars_detected)
            self.modify_box_coordinates(pillars_detected)

        return pillars_detected

    def determine_object_class(self, pillars_detected):
        """
        Method naming all pillars detected
        :param pillars_detected:
        :return:
        """
        # There is only one pillar and each window (pole image section).
        for window, pillars in pillars_detected.items():
            for pillar in pillars:
                pillar.object_name = "pillar"

    def modify_box_coordinates(self, pillars_detected):
        """
        Modifies pillars coordinates (cuts off the bottom part to try to eliminate any
        other objects like tree tops etc that can cause problems when it comes to finding
        the pole's edge via Canny edge detection and HoughLinesP
        :param pillars_detected:
        :return:
        """
        for image_section, pillar in pillars_detected.items():
            # Since it is list in the list
            pillar = pillar[0]
            new_bot_boundary = int(pillar.BB_bottom * 0.5)
            pillar.update_object_coordinates(bottom=new_bot_boundary)
