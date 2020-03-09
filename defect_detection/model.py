from neural_networks import PolesDetector, ComponentsDetector
from neural_networks import YOLOv3
from defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from utils import ResultsHandler, MetaDataExtractor
from utils import FrameReader, FrameWriter, FrameDisplayer, GetFrame
from collections import defaultdict
from imutils.video import FPS
from queue import Queue
from threading import Thread
import cv2
import time
import sys
import os
import time


class MainDetector:
    """

    """
    def __init__(
            self,
            save_path: str,
            crop_path: str=None,
            search_defects: bool=True
    ):

        self.save_path = save_path
        self.crop_path = crop_path

        if search_defects:
            self.defects = True
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

        #self.Q_to_block_1 = Queue(maxsize=12)

    def predict(
            self,
            path_to_data: str,
            pole_number: int=None
    ) -> dict:
        """
        :param path_to_data: Path to data from an API call
        :param pole_number: Number of the pole to which the data being processed belongs. Used to
        comply with the naming convention for saving the processed data
        :return:
        """
        detected_defects = {}

        # API call sends a link to data on the server to process. It can be an image, a video or
        # a folder with image(s) and video(s)
        if os.path.isfile(path_to_data):

            # Find out if this is a video or an image and preprocess it accordingly
            item_name = os.path.basename(path_to_data)

            # TODO: We might not be able to process PNGs. Double check

            if any(item_name.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):

                defects = self.search_defects(path_to_image=path_to_data,
                                              pole_number=pole_number)

                detected_defects[item_name] = defects

            elif any(item_name.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):

                defects = self.search_defects(path_to_video=path_to_data,
                                              pole_number=pole_number)

                detected_defects[item_name] = defects

            else:
                raise TypeError("ERROR: File's extension cannot be processed")

        elif os.path.isdir(path_to_data):
            # This is a folder.

            # How do we keep store multiple JSON files from each processed object? We need some sort
            # of container - a dictionary.
            time_taken = []
            N_of_files = len(os.listdir(path_to_data))

            for item in os.listdir(path_to_data):

                # TODO: We might not be able to process PNGs. Double check

                if any(item.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):

                    print("\nImage:", item)
                    path_to_image = os.path.join(path_to_data, item)

                    t = time.time()
                    defects = self.search_defects(path_to_image=path_to_image,
                                                  pole_number=pole_number)
                    time_taken.append((item, time.time() - t))

                    # If successfully processed, store defects found
                    if defects:
                        detected_defects[item] = defects
                    else:
                        detected_defects[item] = {}

                elif any(item.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):

                    print("Video:", item)
                    path_to_video = os.path.join(path_to_data, item)

                    defects = self.search_defects(path_to_video=path_to_video,
                                                  pole_number=pole_number)

                    # If successfully processed, store defects found
                    if defects:
                        detected_defects[item] = defects
                    else:
                        detected_defects[item] = {}

                # TO CONFIRM: We could potentially add processing of sub-folder via recursion.
                elif os.path.isdir(os.path.join(path_to_data, item)):
                    continue

                else:
                    continue

        else:
            raise TypeError("ERROR: Wrong input. Neither folder nor file")

        # total_time = sum(time for filename, time in time_taken)
        # time_taken.sort(key=lambda e:e[1], reverse=True)
        # #longest_processing = max(time_taken, key=lambda e: e[1])
        # print("\nAVG Time:", round(total_time/N_of_files, 3))
        # print("\nLongest to process:")
        # for e in time_taken:
        #     print(e)

        print("\nDETECTED DEFECTS:")
        for k, v in detected_defects.items():
            print(k, v)

        return detected_defects

    def search_defects(
            self,
            path_to_image: str=None,
            path_to_video: str=None,
            pole_number: int=None
    ) -> dict:
        """
        :param path_to_image:
        :param path_to_video:
        :param pole_number:
        :return:
        """
        # If image, just read it.
        if path_to_image:
            # try:
            #     frame = cv2.imread(path_to_image)
            # except:
            #     print("Failed to open:", os.path.basename(path_to_image))
            #     return {}

            # video_stream = FrameReader(path=path_to_image, Q=self.Q_to_block_1)
            # video_stream.daemon = True
            # video_stream.start()
            # time.sleep(1)

            video_stream = GetFrame(path=path_to_image)

            # Here we can potentially check image metadata -> tuple: (pitch_angle, roll_angle)
            camera_orientation = self.meta_data_extractor.get_angles(path_to_image)
            filename = os.path.basename(path_to_image)

        else:
            filename = os.path.basename(path_to_video)
            camera_orientation = (0, 0)

            #Launch a thread for reading video frames
            # video_stream = FrameReader(path=path_to_video, Q=self.Q_to_block_1)
            # video_stream.start()
            # time.sleep(1)

            video_stream = GetFrame(path=path_to_video)
            video_stream.start()

        video_writer = None

        # Keep track of detected defects (keys - video frames, values - dict of defects)
        defects = {}

        # Run inference once in N frames
        frame_counter = 0
        fps = FPS().start()

        while cv2.waitKey(1) < 0:

            if video_stream.done:
                video_stream.stop()
                break

            image_to_process = video_stream.frame
            if path_to_image:
                video_stream.done = True

            #image_to_process = video_stream.get_frame()

            if video_writer is None and path_to_video:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(os.path.join(self.save_path, 'video_out.avi'), fourcc, 30,
                                              (image_to_process.shape[1],
                                               image_to_process.shape[0]), True)

            # To keep track of all objects detected by the neural networks
            detected_objects = dict()

            # Process 1 in N frames to increase performance speed
            # if frame_counter % 5 != 0:
            #             #     frame_counter += 1
            #             #     # TODO: Extrapolate BBs. Draw old coordinates. Call Results Handler
            #             #     continue

            # OBJECT DETECTION: Detect and classify poles on the image_to_process
            poles = self.pole_detector.predict(image_to_process)

            # Detect components on each pole detected (insulators, dumpers, concrete pillars)
            components = self.component_detector.predict(image_to_process, poles)

            #DEFECT DETECTION
            defect_time = time.time()
            if self.defects and components:
                start = time.time()
                detected_defects = self.defect_detector.search_defects(detected_objects=components,
                                                                       camera_orientation=camera_orientation,
                                                                       pole_number=pole_number,
                                                                       image_name=filename)
                defect_time = time.time() - start

                # Could potentially use pole number here
                defects[frame_counter] = detected_defects

            # Combine all objects detected into one dict for further processing
            for d in (poles, components):
                detected_objects.update(d)

            # TO DO: Send pole number for post processing

            # Process the objects detected
            # if self.crop_path:
            #     self.handler.save_objects_detected(image=image_to_process,
            #                                        objects_detected=detected_objects,
            #                                        video_writer=video_writer,
            #                                        frame_counter=frame_counter,
            #                                        image_name=filename)

            self.handler.draw_bounding_boxes(objects_detected=detected_objects,
                                             image=image_to_process)

            self.handler.save_frame(image=image_to_process,
                                    image_name=filename,
                                    video_writer=video_writer)

            #cv2.imshow("Frame", image_to_process)

            frame_counter += 1

            print(f"Defects: {round(defect_time, 3)}")

            fps.update()

        fps.stop()
        #print("Elapsed time: {:.2f}".format(fps.elapsed()))
        #print("Approx FPS: {:.2f}".format(fps.fps()))

        video_stream.stop()
        cv2.destroyAllWindows()

        return defects

if __name__ == "__main__":

    SAVE_PATH = r"D:\Desktop\system_output\RESULTS"
    #PATH_TO_DATA = r"D:\Desktop\system_output\TEST_IMAGES\IMG_3022.JPG"
    #PATH_TO_DATA = r"D:\Desktop\Reserve_NNs\Datasets\raw_data\Utility Poles\more_tower_photos\anchor\002.PNG"
    PATH_TO_DATA = r"D:\Desktop\system_output\FINAL_TILT_TEST\DJI_0109_1950.jpg"
    #PATH_TO_DATA = r"D:\Desktop\Reserve_NNs\Datasets\raw_data\videos_Oleg\Some_Videos\isolators\DJI_0306.MP4"
    #PATH_TO_DATA = r'D:\Desktop\system_output\TEST_concrete_only'

    pole_number = 123

    detector = MainDetector(save_path=SAVE_PATH)

    defects = detector.predict(path_to_data=PATH_TO_DATA,
                               pole_number=pole_number)
