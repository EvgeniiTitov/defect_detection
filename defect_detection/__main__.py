from neural_networks import ResultsHandler, PoleDetector, ComponentsDetector, PillarDetector
from neural_networks import NetPoles, NetElements, NetPillars
from defect_detectors import DefectDetector, MetaDataExtractor
import cv2
import time
import sys
import os
import argparse


class Detector:

    def __init__(self,
                 save_path,
                 crop_path=None,
                 defects=None):

        self.save_path = save_path
        self.crop_path = crop_path

        # Initialize defect detector and check what defects need to be checked for
        self.detect_pole_defects = False
        self.detect_dumper_defects = False
        self.detect_insulator_defects = False
        if defects:
            self.defect_detector = DefectDetector(defects)
            for component, detecting_flag in defects.items():
                if detecting_flag and component == "pole_defects":
                    self.detect_pole_defects = True
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
        # (represent them in a convenient way we wish, modify BBs etc.)
        self.pole_detector = PoleDetector(self.poles_neuralnet)
        self.component_detector = ComponentsDetector(self.components_neuralnet)
        self.pillars_detector = PillarDetector(self.pillars_neuralnet)

        # Initialize results handler that shows/saves detection results
        self.handler = ResultsHandler(save_path=self.save_path,
                                      cropped_path=self.crop_path)

        # Set up a window
        self.window_name = "Defect Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def predict(self,
               path_to_input,
               image=None,
               folder=None,
               video=None):
        """
        Main function that gets fed an input file (image, folder of images, video). It processes
        the input.
        :param path_to_input: Path to an input file regardless of its nature (image,folder,video)
        :param image: Flag indicating the input type
        :param folder: Flag indicating the input type
        :param video: Flag indicating the input type
        :return: 1 once the input's been processed
        """
        metadata = None

        if all((image, path_to_input)):
            image_name = os.path.split(path_to_input)[-1].split('.')[0]
            img = cv2.imread(path_to_input)

            # Get metadata required for pole tilt detection
            if self.detect_pole_defects:
                # tuple (pitch_angle, roll_angle)
                metadata = self.meta_data_extractor.get_error_values(path_to_input)
                if not metadata:
                    print("Image", image_name, " has got no metadata. Cannot check for tilt")

            self.process(image=img,
                         image_name=image_name,
                         metadata=metadata)

        elif all((folder, path_to_input)):
            for file in os.listdir(path_to_input):
                if not any(file.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                    continue
                image_name = file.split('.')[0]
                path_to_image = os.path.join(path_to_input, file)
                img = cv2.imread(path_to_image)

                # Get metadata required for pole tilt detection
                if self.detect_pole_defects:
                    # tuple (pitch_angle, roll_angle)
                    metadata = self.meta_data_extractor.get_error_values(path_to_image)
                    if not metadata:
                        print("Image", image_name, " has got no metadata. Cannot check for tilt")

                self.process(image=img,
                             image_name=image_name,
                             metadata=metadata)

        elif all((video, path_to_input)):
            cap = cv2.VideoCapture(path_to_input)
            video_name = os.path.split(path_to_input)[-1].split('.')[0]
            output_name = os.path.join(self.save_path, video_name + "_out.avi")
            video_writer = cv2.VideoWriter(output_name,
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                          (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            self.process(cap=cap,
                         video_writer=video_writer)

        else:
            print("ERROR: Incorrect input")
            sys.exit()

        return 1

    def process(self,
                image=None,
                metadata=None,
                cap=None,
                video_writer=None,
                image_name=None):
        """
        Function that performs processing of a frame/image. Works both with videos and photos.
        :param image:
        :param cap:
        :param video_writer:
        :return:
        """
        frame_counter = 0
        # Start a loop to process all video frames, for images just one run through
        while cv2.waitKey(1)<0:
            start_time = time.time()
            objects_detected = dict()

            if all((cap, video_writer)):
                has_frame, frame = cap.read()
                if not has_frame:
                    return
            else:
                frame = image

            # TRICK TO INCREASE VIDEO PROCESSING SPEED. Process 1 in N frames
            if frame_counter % 5 != 0:
                frame_counter += 1
                continue

            # Detect and classify poles on the frame
            poles = self.pole_detector.predict(frame)

            # -------------------------------------------------------------------------
            # HERE IT'D BE NICE TO RUN PILLARS AND COMPONENTS DETECTION IN PARALLEL
            # Detect pillars on concrete poles
            pillars = self.pillars_detector.predict(frame, poles)
            # Detect components on each pole detected
            components = self.component_detector.predict(frame, poles)
            # -------------------------------------------------------------------------

            if self.detect_pole_defects and pillars:
                # For now just draw a line on which the decision will be made
                the_line = self.defect_detector.find_defects_pillars(pillars, frame, metadata)
                if the_line:
                    self.handler.draw_the_line(frame, the_line)

            elif self.detect_dumper_defects and components:
                dumper_defects = self.defect_detector.find_defects_dumpers(components, frame)

            elif self.detect_insulator_defects and components:
                insulator_defects = self.defect_detector.find_defects_insulators(components, frame)


            # Combine all objects detected into one dict for further processing
            for d in (poles, components, pillars):
                objects_detected.update(d)

            # Process the objects detected
            if self.crop_path:
                self.handler.save_objects_detected(image=frame,
                                                   objects_detected=objects_detected,
                                                   video_writer=video_writer,
                                                   frame_counter=frame_counter,
                                                   image_name=image_name)
            self.handler.draw_bounding_boxes(objects_detected=objects_detected,
                                             image=frame)
            self.handler.save_frame(image=frame,
                                    image_name=image_name,
                                    video_writer=video_writer)

            cv2.imshow(self.window_name, frame)
            frame_counter += 1
            print("Time taken:", time.time() - start_time)

            if not image is None:
                return


def parse_args():
    parser = argparse.ArgumentParser()

    # Type of input data
    parser.add_argument('--image', type=str,  help='Path to an image.')
    parser.add_argument('--video', type=str, help='Path to a video.')
    parser.add_argument('--folder', type=str, help='Path to a folder containing images.')

    # Managing results
    parser.add_argument('--crop_path', type=str, default=None,
                        help='Path to crop out and save objects detected')
    parser.add_argument('--save_path', type=str, default=r'D:\Desktop\system_output',
                        help="Path to where save images afterwards")
    # Defects to detect
    parser.add_argument('--pole_defects', default=None, help='Perform defect detection on any poles detected')
    parser.add_argument('--dumper_defects', default=None, help='Perform defect detection on any dumpers detected')
    parser.add_argument('--insulator_defects', default=None, help='Perform defect detection on any insulators detected')
    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    arguments = parse_args()

    if not any((arguments.image, arguments.video, arguments.folder)):
        print("You have not provided a single source of data. Try again")
        sys.exit()

    save_path = arguments.save_path

    crop_path = None
    if arguments.crop_path:
        crop_path = arguments.crop_path

    # Check what defects user wants to detect
    defects_to_find = {'pole_defects': 0,
                       'dumper_defects': 0,
                       'insulator_defects': 0}

    if arguments.pole_defects:
        defects_to_find['pole_defects'] = 1
    elif arguments.dumper_defects:
        defects_to_find['dumper_defects'] = 1
    elif arguments.insulator_defects:
        defects_to_find['insulator_defects'] = 1

    detector = Detector(save_path=save_path,
                        crop_path=crop_path,
                        defects=defects_to_find)

    # Check what user has provided as an input
    if arguments.image:
        if not os.path.isfile(arguments.image):
            print("The provided file is not an image")
            sys.exit()
        detector.predict(image=1,
                         path_to_input=arguments.image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()
        detector.predict(folder=1,
                         path_to_input=arguments.folder)

    elif arguments.video:
        if not os.path.isfile(arguments.video):
            print("The provided file is not a video")
            sys.exit()
        start_time = time.time()
        detector.predict(video=1,
                         path_to_input=arguments.video)
        end_time = time.time()
        print("Video has been processed in:", end_time - start_time, " seconds")
