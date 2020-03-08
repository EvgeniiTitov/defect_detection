from neural_networks import PolesDetector, ComponentsDetector
from neural_networks import YOLOv3
from defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from utils import ResultsHandler, MetaDataExtractor
from collections import defaultdict
import cv2
import time
import sys
import os
import argparse


class MainDetector:
    """

    """
    def __init__(
            self,
            save_path,
            crop_path=None,
            defects=True
    ):

        self._performance_tracking = True

        self.save_path = save_path
        self.crop_path = crop_path

        if defects:
            self.defects = defects
            # Implement metadata check here in the main class because otherwise we
            # will need to open the same image multiple times (not efficient). Check
            # happens right before sending image along the pipeline
            self.meta_data_extractor = MetaDataExtractor()
        else:
            self.defects = False

        # Initialize predicting neural nets
        # self.poles_neuralnet = NetPoles()
        poles_network = YOLOv3()
        components_network = YOLOv3()
        pillars_network = YOLOv3()

        # Initialize detectors using the nets above to predict and postprocess the predictions
        # such as represent objects detected in the way we need, modify BBs etc.
        self.pole_detector = PolesDetector(detector=poles_network)
        self.component_detector = ComponentsDetector(components_predictor=components_network,
                                                     pillar_predictor=pillars_network)

        self.defect_detector = DefectDetector(line_modifier=LineModifier,
                                              concrete_extractor=ConcreteExtractor,
                                              cracks_detector=None,
                                              dumpers_defect_detector=None,
                                              insulators_defect_detector=None)

        # Initialize results handler that shows/saves/transforms into JSON detection results
        self.handler = ResultsHandler(save_path=self.save_path,
                                      cropped_path=self.crop_path)

        # Set up a window
        # TO DO: Move it to a separate class in utils
        self.window_name = "Defect Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def process_data(
            self,
            path_to_data,
            pole_number=None
    ):
        """

        :param path_to_data: Path to data from an API call
        :param pole_number: Number of the pole to which the data being processed belongs. Used to
        comply with the naming convention for saving the processed data
        :return:
        """
        # API call sends a link to data on the server to process. It can be an image, a video or
        # a folder with image(s) and video(s)
        if os.path.isfile(path_to_data):
            # Find out if this is a video or an image and preprocess it accordingly
            item_name = os.path.basename(path_to_data)

            if any(item_name.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):
                # This is an image
                if_processed, defects = self.process_image(path_to_image=path_to_data,
                                                           pole_number=pole_number)

                if if_processed:
                    return defects
                else:
                    return None

            elif any(item_name.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):
                # This is a video
                if_processed, defects = self.process_video(path_to_video=path_to_data,
                                                           pole_number=pole_number)

                if if_processed:
                    return defects
                else:
                    return None

            else:
                raise TypeError("ERROR: File's extension cannot be processed")

        elif os.path.isdir(path_to_data):
            # This is a folder.
            # IT IS ASSUMED THERE CAN BE NO SUBFOLDERS. CLARIFY

            # How do we keep store multiple JSON files from each processed object? We need some sort
            # of container - a dictionary.
            processing_results = defaultdict(list)

            for item in os.listdir(path_to_data):

                if any(item.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):

                    print("Image:", item)
                    path_to_image = os.path.join(path_to_data, item)
                    if_processed, defects = self.process_image(path_to_image=path_to_image,
                                                               pole_number=pole_number)

                    # If successfully processed, store defects found
                    if if_processed:
                        processing_results[item].append(defects)
                    else:
                        processing_results[item].append(dict())

                elif any(item.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):

                    print("Video:", item)
                    path_to_video = os.path.join(path_to_data, item)
                    if_processed, defects = self.process_video(path_to_video=path_to_video,
                                                               pole_number=pole_number)

                    # If successfully processed, store defects found
                    if if_processed:
                        processing_results[item].append(defects)
                    else:
                        processing_results[item].append(dict())

                # TO CONFIRM: We could potentially add processing of sub-folder via recursion.
                elif os.path.isdir(os.path.join(path_to_data, item)):
                    continue

                else:
                    continue

        else:
            raise TypeError("ERROR: Wrong input. Neither folder nor file")

    def process_image(
            self,
            path_to_image,
            pole_number
    ):
        """

        :param path_to_image: path to image to process
        :param pole_number: number of the pole to which the image getting processed belongs
        :return: flag (whether was processed successfully), defects if any found
        """
        # Get image's name without its extension and open the image
        image_name = os.path.splitext(os.path.basename(path_to_image))[0]

        try:
            image = cv2.imread(path_to_image)
        except:
            print("Failed to open:", image_name)

            return 0, None

        if self.defects:
            # Check if there is any metadata associated with an image (to check for camera orientation angles for
            # pole inclination detection module). camera_inclination = (pitch_angle, roll_angle)
            camera_inclination = self.meta_data_extractor.get_angles(path_to_image)
        else:
            camera_inclination = None

        defects = self.search_defects(image=image,
                                      image_name=image_name,
                                      camera_orientation=camera_inclination,
                                      pole_number=pole_number)

        return 1, defects

    def process_video(
            self,
            path_to_video,
            pole_number
    ):
        """

        :param path_to_video: path to video to process
        :param pole_number: number of the pole to which the video getting processed belongs
        :return: flag, defects
        """
        video_name = os.path.splitext(os.path.basename(path_to_video))[0]

        # Create cap object containing all video's frames
        try:
            cap = cv2.VideoCapture(path_to_video)
        except:
            print("Failed to create a CAP object for:", video_name)

            return 0, None

        output_name = os.path.join(self.save_path, video_name + "_out.avi")

        # Initialize video writer to reconstruct the video once each frame's been processed
        video_writter = cv2.VideoWriter(
                output_name,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                10,
                (
                    round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                                        )

        defects = self.search_defects(cap=cap,
                                      video_writer=video_writter,
                                      pole_number=pole_number)

        return 1, defects

    def search_defects(
            self,
            image=None,
            camera_orientation=None,
            cap=None,
            video_writer=None,
            image_name=None,
            pole_number=None
    ):
        """
        Function that performs processing of a video or an image. By default all arguments are set to
        None because videos and images require completely different arguments to be sent to this method
        :param image: numpy array (image read with cv2.imread())
        :param camera_orientation: camera orientation angles when the photo was taken
        :param cap: a class object containing all video's frames
        :param video_writer: a class object for writing the video
        :param image_name: image name without its extension
        :return: JSON file with all defects found
        """
        # Run object detection using networks once in N frames
        frame_counter_object_detection = 0

        # Run object detection AND defect detection once in M frames
        frame_counter_defect_detection = 0

        # TO DO:
        # - FRAME TO FRAME TRACKING FOR VIDEOS TO ENSURE SAME DEFECTS APPEAR ON MULTIPLE FRAMES
        # - TRACK TIME FOR EACH STEP AND SEE WHAT TAKES MOST OF IT. OPTIMIZE

        # Start a loop to process all video frames, for images just one run through
        while cv2.waitKey(1) < 0:

            if cap and video_writer:
                has_frame, image_to_process = cap.read()
                # If all video frames have been processed, return
                if not has_frame:
                    return
            else:
                image_to_process = image

            # To keep track of all objects detected by the neural networks
            all_detected_objects = dict()

            # TEMPORARY: Process 1 in N frames to increase performance speed
            # TO CONSIDER: You can get total N of frames in advance. Might be useful
            if frame_counter_object_detection % 7 != 0:
                frame_counter_object_detection += 1
                continue

            # OBJECT DETECTION: Detect and classify poles on the image_to_process
            start = time.time()
            poles = self.pole_detector.predict(image_to_process)
            poles_time = time.time() - start

            # Detect components on each pole detected (insulators, dumpers, concrete pillars)
            start = time.time()
            components = self.component_detector.predict(image_to_process, poles)
            components_time = time.time() - start

            # DEFECT DETECTION
            defect_time = None
            if self.defects and components:
                start = time.time()
                detected_defects = self.defect_detector.search_defects(detected_objects=components,
                                                                        camera_orientation=camera_orientation,
                                                                        pole_number=pole_number,
                                                                        image_name=image_name)
                defect_time = time.time() - start

            # STORE ALL DEFECTS FOUND IN ONE PLACE. JSON?
            # Photo name (ideally pole's number) -> all elements detected -> defect on those elements
            # Video name (ideally pole's number) -> same


            # Combine all objects detected into one dict for further processing
            for d in (poles, components):
                all_detected_objects.update(d)

            # TO DO: Send pole number for post processing

            # Process the objects detected
            if self.crop_path:
                self.handler.save_objects_detected(image=image_to_process,
                                                   objects_detected=all_detected_objects,
                                                   video_writer=video_writer,
                                                   frame_counter=frame_counter_object_detection,
                                                   image_name=image_name)

            self.handler.draw_bounding_boxes(objects_detected=all_detected_objects,
                                             image=image_to_process)

            # self.handler.save_frame(image=image_to_process,
            #                         image_name=image_name,
            #                         video_writer=video_writer)

            cv2.imshow(self.window_name, image_to_process)

            frame_counter_object_detection += 1
            frame_counter_defect_detection += 1

            print(f"Time taken. Poles {round(poles_time, 3)}, C"
                  f"omponents: {round(components_time, 3)}, "
                  f"Defects: {defect_time}")
            print()

            # Break out of the while loop in case we are dealing with an image.
            if image is not None:
                return


def parse_args():
    parser = argparse.ArgumentParser()

    # Type of input data
    parser.add_argument('--image', type=str,  help='Path to an image.')
    parser.add_argument('--video', type=str, help='Path to a video.')
    parser.add_argument('--folder', type=str, help='Path to a folder containing images or videos to process.')

    # Managing results
    parser.add_argument('--crop_path', type=str, default=None,
                        help='Path to crop out and save objects detected')
    parser.add_argument('--save_path', type=str, default=r'D:\Desktop\system_output',
                        help="Path to save input after its been processed")
    # Defects to detect
    parser.add_argument('--concrete_pole_defects', type=int, default=0,
                        help='Perform defect detection on any poles detected')
    parser.add_argument('--dumper_defects', type=int, default=0,
                        help='Perform defect detection on any dumpers detected')
    parser.add_argument('--insulator_defects', type=int, default=0,
                        help='Perform defect detection on any insulators detected')

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":

    SAVE_PATH = r"D:\Desktop\Reserve_NNs\DEVELOPMENT\cracks_for_testing\results"
    #PATH_TO_DATA = r"D:\Desktop\system_output\TEST_IMAGES\DJI_0110_800.JPG"
    #PATH_TO_DATA = r"D:\Desktop\Reserve_NNs\DEVELOPMENT\cracks_for_testing\cracks\00027.jpg"
    PATH_TO_DATA = r"D:\Desktop\Reserve_NNs\Datasets\raw_data\videos_Oleg\Some_Videos\isolators\DJI_0306.MP4"
    #PATH_TO_DATA = r'D:\Desktop\Reserve_NNs\DEVELOPMENT\cracks_for_testing\cracks'

    pole_number = 123

    detector = MainDetector(save_path=SAVE_PATH)

    defects = detector.process_data(path_to_data=PATH_TO_DATA,
                                    pole_number=pole_number)
