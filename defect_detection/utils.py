from queue import Queue
import threading
from threading import Thread
import cv2
import os
import numpy as np
import re


class FrameReader(threading.Thread):
    """
    Thread tasked with reading frames and adding them to the Q
    """
    def __init__(
            self,
            path: str,
            Q,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.done = False

        try:
            self.stream = cv2.VideoCapture(path)
        except:
            print("Failed to open the cap")
            self.Q.put("END")

    def run(self) -> None:

        while True:
            if self.done:
                break

            if not self.Q.full():
                has_frame, frame = self.stream.read()

                if not has_frame:
                    self.stop()
                    break

                self.Q.put(frame)

        self.stream.release()
        self.Q.put("END")

    def get_frame(self):
        """
        Get a frame from the Q
        :return:
        """
        return self.Q.get()

    def has_frames(self):
        """
        Check if there're decoded frame in the Q ready to go
        :return:
        """
        return self.Q.qsize() > 0

    def stop(self):
        self.done = True


class GetFrame(object):
    """
    Decodes a single frame in advance
    """
    def __init__(
            self,
            path: str
    ):
        self.stream = cv2.VideoCapture(path)

        if not self.stream.isOpened():
            print("\nERROR: Failed to open the cap")
            return

        # Read the first frame
        self.frame_ready, self.frame = self.stream.read()

        self.done = False

    def decode_frame(self):
        """
        :return:
        """
        while not self.done:

            if not self.frame_ready:
                self.stop()

            else:
                self.frame_ready, self.frame = self.stream.read()

    def stop(self):
        self.done = True

    def start(self):
        Thread(target=self.decode_frame, args=()).start()


class FrameWriter(object):
    """
    Thread tasked with:
    1. Drawing BBs
    2. Saving frames back on the disk
    3. Constructing JSON file if processing photo / video's over
    """
    def __init__(self):

        self.frame = None
        self.done = False
        # Extra attributes to use during processing the results
        # 1. BBs coordinates to draw
        # CAN WE CREATE INSTANCE OF RESULTS HANDLER HERE?

    def process_results(self):
        """

        :return:
        """

        #TODO: Main logic here
        pass

    def stop(self):

        self.done = True

    def start(self):
        """

        :return:
        """
        thread = Thread(target=self.process_results, args=())
        # Runs in the background
        thread.daemon = True
        thread.start()

        return self


class FrameDisplayer(object):
    """

    """
    def __int__(
            self,
            frame: np.ndarray=None
    ):

        self.frame = frame
        self.done = False


    def show(self):
        """

        :return:
        """
        while not self.done:

            cv2.imshow("Frame", self.frame)

            if cv2.waitKey(1) == ord("q"):
                self.stop()

    def stop(self):

        self.done = True

    def start_thread(self):
        """

        :return:
        """
        thread = Thread(target=self.show, args=())
        # Runs in the background
        thread.daemon = True
        thread.start()

        return self


class ResultsHandler:
    """
    Class performing BBs drawing, saving objects to disk
    """
    def __init__(
            self,
            save_path,
            cropped_path
    ):
        self.save_path = save_path
        self.cropped_path = cropped_path

    def line_text_size(self, image):
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
            objects_detected,
            image
    ):
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

    def draw_the_line(self, image, line, tilt_angle):
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


class MetaDataExtractor:

    def __init__(
            self,
            kernel_size=9
    ):
        self.kernel_size = kernel_size

    def get_angles(self, path_to_image):
        """
        Extracts metadata if any regarding camera's orientation when an image was taken
        :return:
        """
        with open(path_to_image, encoding="utf8", errors="ignore") as d:
            metadata = d.read()

            if not metadata:
                return None

            start = metadata.find("<x:xmpmeta")
            end = metadata.find("</x:xmpmeta")
            data = metadata[start:end + 1]

            return self.calculate_error(data)

    def calculate_error(self, metadata):
        """
        Calculates orientation errors (pitch, roll) based on the metadata extracted
        :return: pitch, roll values in tuple
        """
        pitch = str(re.findall(r"drone-dji:FlightPitchDegree=\D\+?\-?\d+\.\d+\D", metadata))
        roll = str(re.findall(r"drone-dji:GimbalRollDegree=\D\+?\-?\d+\.\d+\D", metadata))

        pitch_angle = re.findall(r"\d+.\d+", pitch)
        roll_degree = re.findall(r"\d+.\d+", roll)

        if any((pitch_angle, roll_degree)):
            # Since values are stored in lists
            return float(pitch_angle[0]), float(roll_degree[0])
        else:
            return None
