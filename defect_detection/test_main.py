from .neural_networks.predictors_test import ResultsHandler, PoleDetector, ComponentsDetector, DefectDetector
from .neural_networks import NetPoles, NetElements
import cv2
import time
import sys
import os
import argparse

class Detector:

    def __init__(self,
                 save_path,
                 crop_path=None
                 ):

        self.save_path = save_path
        self.crop_path = crop_path
        # Initialize predicting neural nets
        self.poles_neuralnet = NetPoles()
        self.components_neuralnet = NetElements()
        # Initialize detectors using nets to predict and postprocess the predictions
        self.pole_detector = PoleDetector(self.poles_neuralnet)
        self.component_detector = ComponentsDetector(self.components_neuralnet)
        # TBC. Initialize defect detector
        self.defect_detector = DefectDetector()
        # Initialize results handler that shows/saves detection results
        self.handler = ResultsHandler(save_path=self.save_path,
                                      cropped_path=self.crop_path)

        self.window_name = "Defect Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def predict(self, image):
        start_time = time.time()
        objects_detected = dict()

        poles = self.pole_detector.predict(image)
        components = self.component_detector.predict(image, poles)

        for d in (poles, components):
            objects_detected.update(d)

        self.process_results(objects_detected)

    def process_results(self, objects_detected):
        if self.crop_path:
            self.handler.save_objects_detected(objects_detected)
        self.handler.draw_bounding_boxes(objects_detected)
        self.handler.save_frame()




def parse_args():
    parser = argparse.ArgumentParser()
    # Type of input data
    parser.add_argument('--image', type=str,  help='Path to an image.')
    parser.add_argument('--video', type=str, help='Path to a video.')
    parser.add_argument('--folder', type=str, help='Path to a folder containing images.')
    # Managing results
    parser.add_argument('--crop_path', type=str, default=False,
                        help='Path to crop out and save objects detected')
    parser.add_argument('--save_path', type=str, default=r'D:\Desktop\system_output',
                        help="Path to where save images afterwards")
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

    if arguments.image:
        if not os.path.isfile(arguments.image):
            print("The provided file is not an image")
            sys.exit()
        detector.process_photo(image=arguments.image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()
        detector.process_photo(folder=arguments.folder)

    elif arguments.video:
        if not os.path.isfile(arguments.video):
            print("The provided file is not a video")
            sys.exit()
        detector.process_video(path_to_video=arguments.video)
