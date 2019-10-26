from neural_networks import ResultsHandler, PoleDetector, ComponentsDetector, PillarDetector
from neural_networks import NetPoles, NetElements, NetPillars
from preprocessing import MetaDataExtractor
from defect_detectors import TiltChecker
import cv2
import time
import sys
import os
import argparse


class Detector:

    def __init__(self, save_path,
                 crop_path=None,
                 defects_detection=None):

        self.save_path = save_path
        self.crop_path = crop_path

        # Check if a user wants to perform defect detection
        self.defects_detection = defects_detection
        if self.defects_detection:
            # Initialize defect detector in this case
            self.tilt_checker = TiltChecker()
            # HIDE METADATA EXTRACTOR IN TILT CHECKER SO WE DONT TAKE SPACE HERE
            self.metadata_extractor = MetaDataExtractor()

        # Initialize predicting neural nets
        self.poles_neuralnet = NetPoles()
        self.components_neuralnet = NetElements()
        self.pillars_neuralnet = NetPillars()

        # Initialize detectors using the nets above to predict and postprocess the predictions
        # (represent them in a convenient way we wish)
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
            # Check if user needs this at all
            #metadata = self.metadata_extractor.get_error_values(path_to_input)
            self.process(image=img,
                         metadata=metadata,
                         image_name=image_name)

        elif all((folder, path_to_input)):
            for file in os.listdir(path_to_input):
                if not any(file.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                    continue
                image_name = file.split('.')[0]
                path_to_image = os.path.join(path_to_input, file)
                img = cv2.imread(path_to_image)
                # Check if user needs this at all
                #metadata = self.metadata_extractor.get_error_values(path_to_image)
                self.process(image=img,
                             metadata=metadata,
                             image_name=image_name)

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
                cap=None,
                video_writer=None,
                metadata=None,
                image_name=None):
        """
        Function that performs processing of a frame/image. Works both with videos and photos.
        :param image:
        :param cap:
        :param video_writer:
        :param metadata: camera orientation errors for pole tilt detection
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

            # TRICK TO INCREASE VIDEO PROCESSING SPEED. Process 1 in 15 frames
            if frame_counter % 40 != 0:
                frame_counter += 1
                continue

            # Detect and classify poles on the frame
            poles = self.pole_detector.predict(frame)
            # -------------------------------------------------------------------------
            # HERE IT'D BE NICE TO RUN PILLAR AND COMPONENTS PREDICTION IN PARALLEL
            # Detect pillars on concrete poles
            pillars = self.pillars_detector.predict(frame, poles)

            # Then send those pillars to the TiltChecker class for potential defect detection

            # Detect components on each pole detected
            components = self.component_detector.predict(frame, poles)
            # -------------------------------------------------------------------------

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
    parser.add_argument('--tilt', default=None, help='Perform pole tilting detection')
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

    detector = Detector(save_path=save_path,
                        crop_path=crop_path)

    defects_detection = 0
    if arguments.tilt:
        defects_detection = 1

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
        detector.predict(video=1,
                         path_to_input=arguments.video)
