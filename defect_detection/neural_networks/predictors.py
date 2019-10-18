from collections import defaultdict
from neural_networks.detections import DetectedObject, DetectionImageSection
from neural_networks.models import NetPoles, NetElements
import numpy as np
import cv2
import os


class ComponentsDetector:
    """
    Class performing predictions of utility pole components on image / image
    sections provided.
    All components detected get represented as class objects and are saved in a dictionary
    as values, whereas the image section on which the detection was performed serves the
    role of a dictionary key.
    """
    def __init__(self):
        # Initialize components predictor
        self.components_predictor = NetElements()
        # Dictionary to keep components detected
        self.components_detected = defaultdict(list)

    def determine_object_class(self):
        """
        Checks object's class and names it. Since we've got multiple nets predicting objects
        like 0,1,2 classes, we want to make sure we don't get confused later.
        Dynamically creates new attribute
        :return: Nothing. Simply names objects
        """
        for window, components in self.components_detected.items():
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
                            self.components_detected[pole_image_section].append(
                                DetectedObject(component[0], component[1], component[2],
                                               component[3], component[4], component[5])
                                                                                )
        else:
            # In case no poles have been detected, send the whole image for components detection
            # in case there are any close-up components on the image
            components = self.components_predictor.predict(image)
            if components:
                whole_image = DetectionImageSection(image, "components")
                for component in components:
                    self.components_detected[whole_image].append(
                        DetectedObject(component[0], component[1], component[2],
                                       component[3], component[4], component[5])
                                                                )
        #  Name objects detected by unique names instead of default 0,1,2 etc.
        if components:
            self.determine_object_class()

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

    def determine_object_class(self):
        """
        Checks object's class and names it. Since we've got multiple nets predicting objects
        like 0,1,2 classes, we want to make sure we don't get confused later.
        Dynamically creates new attribute
        :return: Nothing. Simply names objects
        """
        for window, poles in self.poles_detected.items():
            for pole in poles:
                if pole.class_id == 0:
                    pole.object_name = "metal"
                else:
                    pole.object_name = "concrete"

    def modify_box_coordinates(self, image):
        """
        Modifies pole's BB.
        :param image: image on which detection of poles took place (original image).
        Will be used to make sure new modified coordinates do not go beyond image's edges
        :return: None. Simply modified coordinates
        """
        for window, poles in self.poles_detected.items():
            # Let's consider all poles detected on an image and modify their coordinates
            # If only one pole's been detected, just widen the box 50% both sides
            if len(poles) == 1:
                left = int(poles[0].BB_left*0.5)
                right = int(poles[0].BB_right*0.5) if int(poles[0].BB_right*0.5) <\
                                                   image.shape[1] else image.shape[1]
                poles[0].update_object_coordinates(left=left, right=right)
            else:
                for pole in poles:

                    # ! CHECK FOR OVERLAPPING

                    left = int(pole.BB_left*0.8)
                    right = int(pole.BB_right*1.2) if int(pole.BB_right*1.2) <\
                                                    image.shape[1] else image.shape[1]
                    pole.update_object_coordinates(left=left, right=right)

    def predict(self, image):
        """
        :param image: Image on which to perform pole detection (the whole original image)
        :return: Dictionary containing all poles detected on the image
        """
        detecting_image_section = DetectionImageSection(image, "poles")
        poles = self.poles_predictor.predict(image)
        # Represent each object detected as a class object. Add all objects
        # to the dictionary as values.
        # Check if any poles have been detected otherwise return the empty dictionary
        if poles:
            for pole in poles:
                self.poles_detected[detecting_image_section].append(
                        DetectedObject(pole[0], pole[1], pole[2], pole[3], pole[4], pole[5])
                                                                    )
            # Modify poles coordinates to widen them for components detection
            #self.modify_box_coordinates(image)
            # Name objects detected by unique names instead of default 0,1,2 etc.
            self.determine_object_class()

        return self.poles_detected


class ResultsHandler:
    """
    Class performing BBs drawing, saving objects to disk
    """
    def __init__(self,
                image,
                path_to_image,
                save_path=r"D:\Desktop\system_output",
                cropped_path=r"D:\Desktop\system_output\cropped_elements",
                input_photo=True,
                video_writer=None,
                frame_counter=0
                ):
        self.image = image
        self.path_to_image = path_to_image
        self.save_path = save_path
        self.cropped_path = cropped_path
        self.input_photo = input_photo  # True - photo, False - video
        self.video_writer = video_writer
        self.frame_counter = frame_counter
        # Extract image name. Yikes approach because can be both .jpg and .jpeg
        self.image_name = os.path.split(path_to_image)[-1].split('.')[0]

    def line_text_size(self):
        """
        Method determining BB line thickness and text size based on the original image's size
        :return:
        """
        line_thickness = (self.image.shape[0] * self.image.shape[1] // 1_000_000)
        text_size = 0.5 + (self.image.shape[0] * self.image.shape[1] // 5_000_000)
        text_boldness = 1 + (self.image.shape[0] * self.image.shape[1] // 2_000_000)

        return line_thickness, text_size, text_boldness

    def draw_bounding_boxes(self, objects_detected):
        """
        Method drawing BBs of the objects detected on the image
        :param objects_detected: iterable containing all objects detected
        :return: None
        """
        # Traverse over all key-value pairs drawing objects one after another
        for image_section, elements in objects_detected.items():
            # There might be multiple objects detected in a certain image section (whole image:poles),
            # pole1:elements, pole2:elements etc.
            for element in elements:
                cv2.rectangle(self.image, (image_section.left + element.left, image_section.top + element.top),
                                          (image_section.left + element.right, image_section.top + element.bottom),
                                          (0, 255, 0), self.line_text_size()[0])

                label = "{}:{:1.2f}".format(element.object_name, element.confidence)

                label_size, base_line = cv2.getTextSize(label,
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        self.line_text_size()[1], 1)

                top = max(element.top + image_section.top, label_size[1])

                cv2.putText(self.image, label,
                            (element.left + image_section.left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, self.line_text_size()[1],
                            (0, 0, 0), self.line_text_size()[-1])

    def save_objects_detected(self, objects_detected):
        """
        Class method saving objects detected
        :param objects_detected:
        :return:
        """
        for image_section, elements in objects_detected.items():
            for element in elements:
                if self.input_photo:
                    # Processing image
                    cropped_frame = image_section.frame[element.top+image_section.top:
                                                        element.bottom+image_section.top,
                                                        element.left+image_section.left:
                                                        element.right+image_section.left]
                    file_name = self.image_name + "_OBJECT CLASS" + ".jpg"
                    cv2.imwrite(os.path.join(self.cropped_path, file_name), cropped_frame)
                else:
                    # ! NEEDS TESTING Processing video
                    cropped_frame = image_section.frame[element.top + image_section.top:
                                                        element.bottom + image_section.top,
                                                        element.left + image_section.left:
                                                        element.right + image_section.left]
                    frame_name = "TBC"
                    cv2.imwrite(os.path.join(self.cropped_path, frame_name), cropped_frame)

    def save_frame(self):
        """
        Saves a frame with all BBs drawn on it
        :return:
        """
        if self.input_photo:
            image_name = "out_" + os.path.split(self.path_to_image)[-1]
            cv2.imwrite(os.path.join(self.save_path, image_name), self.image)
        else:
            self.video_writer.write(self.image.astype(np.uint8))


class DefectDetector:
    def find_defects(self, objects):
        return objects
