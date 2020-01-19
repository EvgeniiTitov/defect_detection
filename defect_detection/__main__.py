from neural_networks import PoleDetector, ComponentsDetector, PillarDetector
from neural_networks import NetPoles, NetElements, NetPillars
from defect_detectors import DefectDetector
from utils import ResultsHandler, MetaDataExtractor
import cv2
import time
import sys
import os
import argparse


class Detector:
    """

    """
    def __init__(
            self,
            save_path,
            crop_path=None,
            defects=None
    ):

        self.save_path = save_path
        self.crop_path = crop_path

        # Initialize defect detector and check what defects need to be checked for
        self.detect_concrete_pole_defects = False
        self.detect_dumper_defects = False
        self.detect_insulator_defects = False
        if defects:
            self.defect_detector = DefectDetector(defects)

            for component, detecting_flag in defects.items():
                if detecting_flag and component == "concrete_pole_defects":
                    self.detect_concrete_pole_defects = True
                    self.meta_data_extractor = MetaDataExtractor()

                elif detecting_flag and component == "dumper_defects":
                    self.detect_dumper_defects = True

                elif detecting_flag and component == "insulator_defects":
                    self.detect_insulator_defects = True

        # Initialize predicting neural nets
        self.poles_neuralnet = NetPoles()
        self.components_neuralnet = NetElements()
        self.pillars_neuralnet = NetPillars()

        # Initialize detectors using the nets above to predict and postprocess the predictions
        # such as represent objects detected in the way we need, modify BBs etc.
        self.pole_detector = PoleDetector(self.poles_neuralnet)
        self.component_detector = ComponentsDetector(self.components_neuralnet)
        self.pillars_detector = PillarDetector(self.pillars_neuralnet)

        # Initialize results handler that shows/saves detection results
        self.handler = ResultsHandler(save_path=self.save_path,
                                      cropped_path=self.crop_path)

        # Set up a window
        # TO DO: Move it to a separate class in utils
        self.window_name = "Defect Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def process_input_data(self, input_data):
        """
        Main function that receives a dictionary containing all input data from a user (image,
        video, folder (that can contain image(s) and video(s))
        It runs across the dicrionary and depending on object's type if send it to the appropriate
        handling method.
        """
        for input_type, path_to_data in input_data.items():

            if input_type == "image":
                # Check if an image provided is actually of a format that can be opened
                if not any(
                        os.path.split(path_to_data)[-1].endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
                           ):
                    print(f"ERROR: {path_to_data} is not an image. Cant be processed")
                    continue

                # Process an image, check its extension, metadata and send for defect detection
                successfully_processed = self.process_image(path_to_data)

                image_name = os.path.splitext(os.path.basename(path_to_data))[0]
                if successfully_processed:
                    print(f"Image: {image_name}'s been processed and saved to {self.save_path}")
                else:
                    continue

            elif input_type == "video":
                # Check video's extension
                if not any(
                        os.path.split(path_to_data)[-1].endswith(ext) for ext in ["avi", "MP4", "AVI"]
                           ):
                    print(f"ERROR: Cannot open video {path_to_data}. Wrong extension!")
                    continue

                successfully_processed = self.process_video(path_to_data)

                video_name = os.path.splitext(os.path.basename(path_to_data))
                if successfully_processed:
                    print(f"Video: {video_name}'s been processed and saved to {self.save_path}")
                else:
                    continue

            elif input_type == "folder":
                # There might be multiple image(s) or video(s) in a folder provided
                for filename in os.listdir(path_to_data):

                    # Check if a file is an image
                    if any(filename.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):
                        processed = self.process_image(os.path.join(path_to_data, filename))

                        if processed:
                            print(f"Image {filename}'s been processed and saved to {self.save_path}")
                        else:
                            continue

                    # Check if a file is a video
                    if any(filename.endswith(ext) for ext in ["avi", "MP4", "AVI"]):
                        processed = self.process_video(os.path.join(path_to_data, filename))

                        if processed:
                            print(f"Video: {filename}'s been processed and saved to {self.save_path}")

                    # If a file is neither a video nor an image, skip it
                    else:
                        continue
            else:
                print("ERROR: Something went wrong. Wrong input type")
                sys.exit()

    def process_image(self, path_to_image):
        """

        :param path_to_image:
        :return:
        """
        # Get image's name without its extension and open the image
        image_name = os.path.splitext(os.path.basename(path_to_image))[0]

        try:
            image = cv2.imread(path_to_image)
        except:
            print("Failed to open:", image_name)
            return 0

        # Check if there is any metadata associated with an image (to check for camera orientation angles for
        # pole inclination detection module). camera_inclination = (pitch_angle, roll_angle)
        if self.detect_concrete_pole_defects:
            camera_inclination = self.meta_data_extractor.estimate_camera_inclination(path_to_image)
        else:
            camera_inclination = None

        self.search_defects(image=image,
                            image_name=image_name,
                            metadata=camera_inclination)

        return 1

    def process_video(self, path_to_video):
        """

        :param video:
        :return:
        """
        video_name = os.path.splitext(os.path.basename(path_to_video))[0]

        # Create cap object containing all video's frames
        try:
            cap = cv2.VideoCapture(path_to_video)
        except:
            print("Failed to create a CAP object for:", video_name)
            return 0

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

        self.search_defects(cap=cap,
                            video_writer=video_writter)

        return 1

    def search_defects(
            self,
            image=None,
            metadata=None,
            cap=None,
            video_writer=None,
            image_name=None
    ):
        """
        Function that performs processing of a video or an image. By default all arguments are set to
        None because videos and images require completely different arguments to be sent to this method
        :param image: numpy array (image read with cv2.imread())
        :param metadata: camera orientation angles when the photo was taken
        :param cap: a class object containing all video's frames
        :param video_writer: a class object for writing the video
        :param image_name: image name without its extension
        :return:
        """
        frame_counter = 0

        # TO DO:
        # FRAME TO FRAME TRACKING FOR VIDEOS TO ENSURE SAME DEFECTS APPEAR ON MULTIPLE FRAMES

        # Start a loop to process all video frames, for images just one run through
        while cv2.waitKey(1) < 0:
            time_1 = time.time()

            # To keep track of all objects detected by the neural networks
            all_detected_objects = dict()

            if cap and video_writer:
                has_frame, image_to_process = cap.read()
                # If all video frames have been processed, return
                if not has_frame:
                    return
            else:
                image_to_process = image

            # TEMPORARY: Process 1 in N frames to increase performance speed
            if frame_counter % 2 != 0:
                frame_counter += 1
                continue

            # OBJECT DETECTION: Detect and classify poles on the image_to_process
            poles = self.pole_detector.predict(image_to_process)

            # -------------------------------------------------------------------------
            # TEMPORARY: There will be just one component detector, that depending on the tower's class
            # will send it to either concrete or metal neural net

            # Detect pillars on concrete poles
            pillars = self.pillars_detector.predict(image_to_process, poles)
            # Detect components on each pole detected
            components = self.component_detector.predict(image_to_process, poles)
            # -------------------------------------------------------------------------

            # DEFECT DETECTION:
            # if self.detect_concrete_pole_defects and pillars:
            #     # For now just draw a line on which the decision will be made
            #     the_line, tilt_angle = self.defect_detector.find_defects_pillars(pillars, image_to_process, metadata)
            #     if the_line:
            #         self.handler.draw_the_line(image_to_process, the_line, tilt_angle)
            #         #print("Angle:", tilt_angle)
            # elif self.detect_dumper_defects and components:
            #     dumper_defects = self.defect_detector.find_defects_dumpers(components, image_to_process)
            #
            # elif self.detect_insulator_defects and components:
            #     insulator_defects = self.defect_detector.find_defects_insulators(components, image_to_process)

            # STORE ALL DEFECTS FOUND IN ONE PLACE. JSON?
            # Photo name (ideally pole's number) -> all elements detected -> defect on those elements
            # Video name (ideally pole's number) -> same


            # Combine all objects detected into one dict for further processing
            for d in (poles, components, pillars):
                all_detected_objects.update(d)


            # Process the objects detected
            if self.crop_path:
                self.handler.save_objects_detected(image=image_to_process,
                                                   objects_detected=all_detected_objects,
                                                   video_writer=video_writer,
                                                   frame_counter=frame_counter,
                                                   image_name=image_name)

            self.handler.draw_bounding_boxes(objects_detected=all_detected_objects,
                                             image=image_to_process)

            self.handler.save_frame(image=image_to_process,
                                    image_name=image_name,
                                    video_writer=video_writer)

            cv2.imshow(self.window_name, image_to_process)

            frame_counter += 1
            print("Time taken:", time.time() - time_1, "\n")

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
    arguments = parse_args()

    # TO DO: potentially can get more than 1 input, needs to be addressed
    if not any((arguments.image, arguments.video, arguments.folder)):
        print("ERROR: No data to process has been provided")
        sys.exit()

    if not os.path.exists(arguments.save_path):
        os.mkdir(arguments.save_path)
    else:
        if not os.path.isdir(arguments.save_path):
            print("ERROR: Wrong SAVE PATH. It is not a folder")
            sys.exit()

    if arguments.crop_path:
        if not os.path.exists(arguments.crop_path):
            os.mkdir(arguments.crop_path)
        else:
            if not os.path.isdir(arguments.crop_path):
                print("ERROR: Wrong CROP PATH. It is not a folder")
                sys.exit()

    # Check what defects user wants to detect
    defects_to_find = {'concrete_pole_defects': 0,
                       'dumper_defects': 0,
                       'insulator_defects': 0}

    if arguments.concrete_pole_defects:
        defects_to_find['concrete_pole_defects'] = 1
    elif arguments.dumper_defects:
        defects_to_find['dumper_defects'] = 1
    elif arguments.insulator_defects:
        defects_to_find['insulator_defects'] = 1

    detector = Detector(save_path=arguments.save_path,
                        crop_path=arguments.crop_path,
                        defects=defects_to_find)

    # Dictionary to keep track of all the data provided by a user that needs
    # to be processed
    data_to_process = dict()

    if arguments.image:
        if not os.path.isfile(arguments.image):
            print("ERROR: Provided image is not an image")
            sys.exit()

        data_to_process["image"] = arguments.image

    if arguments.video:
        if not os.path.isfile(arguments.video):
            print("ERROR: Provided video is not a video")
            sys.exit()

        data_to_process["video"] = arguments.video

    if arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()

        data_to_process["folder"] = arguments.folder

    start_time = time.time()
    detector.process_input_data(data_to_process)
    time_elapsed = time.time() - start_time

    print("\nAll data's been processed in:", time_elapsed)
