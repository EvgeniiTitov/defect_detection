from neural_networks.predictors import ResultsHandler, PoleDetector, ComponentsDetector, DefectDetector
import cv2
import time
import sys
import os
import argparse

class App:

    def __init__(self,
                 save_path,
                 crop_path=None
                 ):

        self.save_path = save_path
        self.crop_path = crop_path

        self.pole_detector = PoleDetector()
        self.component_detector = ComponentsDetector()
        self.defect_detector = DefectDetector()

        self.handler = ResultsHandler(save_path=self.save_path,
                                      cropped_path=self.crop_path)

        self.window_name = "Defect Detection"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.objects_detected = dict()

    def get_image_paths(self, image=None, folder=None):
        images = list()
        if image:
            images.append(image)
        elif folder:
            for image in os.listdir(folder):
                if not any(image.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                    continue
                # Check for incorrect image names (line spaces, russian letters, prohibited signs!
                # Check with set overlapping?
                images.append(os.path.join(folder, image))

        return images

    def process_photo(self, image=None, folder=None):
        # Process input and collect path(s) to image(s) to process
        images_to_process = []
        if image:
            images_to_process = self.get_image_paths(image=image)
        elif folder:
            images_to_process = self.get_image_paths(folder=folder)

        # Process all images
        for image_path in images_to_process:
            print("Processing:", os.path.split(image_path)[-1])
            start_time = time.time()
            frame = cv2.imread(image_path)

            poles = self.pole_detector.predict(frame)
            components = self.component_detector.predict(frame, poles)

            for d in (poles, components):
                self.objects_detected.update(d)

            if self.crop_path:
                self.handler.save_objects_detected(self.objects_detected)
            self.handler.draw_bounding_boxes(self.objects_detected)
            self.handler.save_frame()

            cv2.imshow(self.window_name, frame)
            end_time = time.time()
            print("Time taken:", end_time - start_time)

    def process_video(self, path_to_video):
        cap = cv2.VideoCapture(path_to_video)
        video_name = os.path.split(path_to_video)[-1][:-4]
        output_name = video_name + "_OUT.avi"
        video_writer = cv2.VideoWriter(output_name,
                                       cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                                       10,
                                       (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                       )

        frame_counter = 0
        while cv2.waitKey(1) < 0:
            start_time = time.time()
            has_frame, frame = cap.read()
            if not has_frame:
                print("Done")
                sys.exit()

            poles = self.pole_detector.predict(frame)
            components = self.component_detector.predict(frame, poles)

            for d in (poles, components):
                self.objects_detected.update(d)

            if self.crop_path:
                self.handler.save_objects_detected(self.objects_detected)
            self.handler.draw_bounding_boxes(self.objects_detected)
            self.handler.save_frame()

            cv2.imshow(self.window_name, frame)
            frame_counter += 1
            end_time = time.time()
            print("Time taken:", end_time - start_time)


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

    app = App(save_path=save_path,
              crop_path=crop_path)

    if arguments.image:
        if not os.path.isfile(arguments.image):
            print("The provided file is not an image")
            sys.exit()
        app.process_photo(image=arguments.image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()
        app.process_photo(folder=arguments.folder)

    elif arguments.video:
        if not os.path.isfile(arguments.video):
            print("The provided file is not a video")
            sys.exit()
        app.process_video(path_to_video=arguments.video)
