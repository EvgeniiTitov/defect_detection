import cv2
import os
import numpy as np
from typing import List
import torch


class ResultsHandler:
    """
    Class performing BBs drawing, saving objects to disk
    """
    def __init__(
            self,
            save_path:str,
            cropped_path:str=None
    ):
        self.save_path = save_path
        self.cropped_path = cropped_path

    def draw_bb_save_image(
            self,
            image: np.ndarray,
            detected_objects: dict,
            pole_number: int,
            image_name: str
    ) -> None:
        """
        TBA
        :param image:
        :param detected_objects:
        :param pole_number:
        :return:
        """
        store_path = os.path.join(self.save_path, str(pole_number))
        if not os.path.exists(store_path):
            os.mkdir(store_path)

        self.draw_bounding_boxes(objects_detected=detected_objects,
                                 image=image)

        new_name = image_name + "_out.jpg"
        cv2.imwrite(os.path.join(store_path, new_name), image)

    def line_text_size(self, image: np.ndarray) -> tuple:
        """
        Method determining BB line thickness and text size based on the original image's size
        :return:
        """
        line_thickness = int(image.shape[0] * image.shape[1] // 1_000_000)
        text_size = 0.5 + (image.shape[0] * image.shape[1] // 5_000_000)
        text_boldness = 1 + (image.shape[0] * image.shape[1] // 2_000_000)

        return line_thickness, text_size, text_boldness

    def draw_bounding_boxes(
            self,
            objects_detected: dict,
            image: np.ndarray
    ) -> None:
        """
        Method drawing BBs of the objects detected on the image
        :param objects_detected: iterable containing all objects detected
        :return: None
        """
        # Traverse over all key-value pairs drawing objects one after another
        colour = (0, 255, 0)
        for image_section, elements in objects_detected.items():
            # There might be multiple objects detected in a certain image section (whole image:poles),
            # pole1:elements, pole2:elements etc.
            for element in elements:

                # Check element class and change BBs colour
                if element.object_name == "insl":
                    colour = (210, 0, 210)
                elif element.object_name == "dump":
                    colour = (255, 0, 0)
                elif element.object_name == "pillar":
                    colour = (0, 128, 255)

                    if element.inclination:
                        text = f"Angle: {element.inclination}"
                        cv2.putText(image, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw BBs using both BBs coordinates and coordinates of the image section relative to the original
                # image in which this object was detected
                cv2.rectangle(image, (image_section.left + element.BB_left, image_section.top + element.BB_top),
                                     (image_section.left + element.BB_right, image_section.top + element.BB_bottom),
                                      colour, self.line_text_size(image)[0])

                label = "{}:{:1.2f}".format(element.object_name, element.confidence)

                label_size, base_line = cv2.getTextSize(label,
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        self.line_text_size(image)[1], 1)

                top = max(element.top + image_section.top, label_size[1])

                cv2.putText(image, label,
                            (element.left + image_section.left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, self.line_text_size(image)[1],
                            (0, 0, 0), self.line_text_size(image)[-1])

    def save_objects_detected(
            self,
            image,
            objects_detected,
            video_writer=None,
            frame_counter=None,
            image_name=None
    ):
        """
        Class method saving objects detected (croping them out)
        :param objects_detected:
        :return:
        """
        for image_section, elements in objects_detected.items():

            # Use enumerate() to make sure no objects get overwritten
            for index, element in enumerate(elements, start=1):
                if not video_writer:
                    # Processing image(s)
                    # There used to be image_section.frame[]
                    cropped_frame = image[element.BB_top + image_section.top:
                                          element.BB_bottom + image_section.top,
                                          element.BB_left + image_section.left:
                                          element.BB_right + image_section.left]

                    file_name = image_name + "_" + element.object_name + "_" + str(index) + ".jpg"

                    cv2.imwrite(os.path.join(self.cropped_path, file_name), cropped_frame)
                else:
                    # ! NEEDS TESTING Processing video
                    cropped_frame = image[element.BB_top + image_section.top:
                                          element.BB_bottom + image_section.top,
                                          element.BB_left + image_section.left:
                                          element.BB_right + image_section.left]

                    frame_name = frame_counter + "_" + element.object_name + "_" + str(index) + ".jpg"

                    cv2.imwrite(os.path.join(self.cropped_path, frame_name), cropped_frame)

    def save_frame(
            self,
            image,
            image_name=None,
            video_writer=None
    ):
        """
        Saves a frame with all BBs drawn on it
        :return:
        """
        if video_writer is None:
            image_name = image_name + "_out.jpg"
            cv2.imwrite(os.path.join(self.save_path, image_name), image)
        else:
            video_writer.write(image.astype(np.uint8))

    def draw_the_line(
            self,
            image,
            line,
            tilt_angle
    ):
        """
        Draws a line which is used for a concrete pole tilt defect detection
        :param image:
        :param line:
        :return:
        """
        # height, width
        label = "Angle: {0:.2f}".format(tilt_angle)
        cv2.putText(image, label, (10, int(image.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX,
                    self.line_text_size(image)[1],
                    (0, 0, 0), self.line_text_size(image)[-1])
        # Line coordinates are relative to the pillar image, not the original one
        cv2.line(image,
                 (line[0], line[1]),
                 (line[2], line[3]),
                 (0, 0, 255), 4,
                 cv2.LINE_AA)

    @staticmethod
    def draw_bb_for_batch(images: List[np.ndarray], batch_predictions: dict) -> None:
        """

        :param image:
        :param batch_predictions:
        :return:
        """
        save_path = r"D:\Desktop\system_output\OUTPUT"
        assert len(images) == len(batch_predictions), "Nb of images in the batch and batch predictions do no match"
        for i in range(len(batch_predictions)):
            image = images[i]
            predictions = batch_predictions[i]

            tower_objects = list(predictions.values())[0]
            for tower_object in tower_objects:
                left = int(tower_object.BB_left)
                top = int(tower_object.BB_top)
                right = int(tower_object.BB_right)
                bot = int(tower_object.BB_bottom)
                #print(f"FROM DRAWER COORDINATES: {left} {top} , {right} {bot}")
                pt1 = left, top
                pt2 = right, bot
                cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

            cv2.imwrite(os.path.join(save_path, f"{i}_out.jpg"), image)

        return
