from collections import defaultdict
from detections import DetectedObject, DetectionImageSection
from neural_nets import NetPoles, NetElements
import numpy as np

# DO NOT FORGET TO PERFORM BBs MODIFICATIONS FOR UTILITY POLES


class ComponentsDetector:
    """
    Class performing predictions of utility pole components on image / image
    sections provided. All components detected get represented as class objects
    and are saved in a dictionary as values, whereas the image section on which the
    detection was performed serves the role of a dictionary key.
    """
    def __init__(self):
        # Initialize components predictor
        self.components_predictor = NetElements()
        # Dictionary to keep components detected
        self.components_detected = defaultdict(list)

    def predict_components(self, image, pole_predictions):
        """
        Predicts components. Saves them in the appropriate format
        :param image: original image in case no poles have been found
        :param pole_predictions: poles predicted by the pole predicting net
        :return: dictionary of components
        """
        components = list()
        # If poles detecting neural net detected any poles. Find all components on them
        if pole_predictions:
            # FOR loop below just to play it safe. There should be only one item in the dictionary
            # original image (class object) : poles detected on it (list of lists)
            for window, poles in pole_predictions.items():
                # Consider all poles detected on the original image
                for pole in poles:
                    # Crop out the pole detected to send for components detection
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
                            self.components_detected[pole_image_section].append(DetectedObject(component))
        else:
            # In case no poles have been detected, send the whole image for components detection
            # in case there are any close-up components on the image
            components = self.components_predictor.predict(image)
            if components:
                whole_image = DetectionImageSection(image, "components")
                for component in components:
                    self.components_detected[whole_image].append(DetectedObject(component))

        return self.components_detected


class PoleDetector:
    """
    Class performing utility poles prediction using the YOLOv3 neural net and
    saving objects detected as class objects in a dictionary for subsequent
    usage.
    Image section on which poles have been detected serves the dictionary's key
    role. In this case we consider the whole image.
    As input it accepts a plain image.
    """
    def __init__(self):
        # Initialize predictor
        self.poles_predictor = NetPoles()
        # Dictionary to keep poles detected
        self.poles_detected = defaultdict(list)

    def modify_box_coordinates(self, pole):
        """
        Modifies pole's BB.
        :param pole: class object
        :return: class object with modified box coordinates
        """
        pass

    def predict(self, image):
        """
        :param image: Image on which to perform pole detection
        :return: Dictionary containing all poles detected on the image
        """
        whole_image = DetectionImageSection(image, "poles")
        poles = self.poles_predictor.predict(image)
        # Represent each object detected as a class object. Add all objects
        # to the dictionary as values.
        # Check if any poles have been detected otherwise return the empty dictionary
        if poles:
            for pole in poles:
                self.poles_detected[whole_image].append(DetectedObject(pole))

        return self.poles_detected



# BBs modification in PolesDetector
