from neural_networks import PolesDetector, ComponentsDetector
from neural_networks import YOLOv3
from defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from utils import ResultsHandler, MetaDataExtractor
from utils import GetFrame
from imutils.video import FPS
import cv2
import os
import time
from collections import defaultdict


class MainDetector:

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

    def predict(
            self,
            path_to_data: str,
            pole_number: int=None
    ) -> dict:
        """
        API endpoint.
        Parses input, for each file(s) provided makes predictions and saves results in dict
        :param path_to_data: Path to data from an API call
        :param pole_number: Number of the pole to which the data being processed belongs. Used to
        comply with the naming convention for saving the processed data
        :return:
        """
        detected_defects = defaultdict(list)

        if os.path.isfile(path_to_data):

            # Find out if this is a video or an image and preprocess it accordingly
            item_name = os.path.basename(path_to_data)

            if any(item_name.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):

                defects = self.search_defects(path_to_image=path_to_data)

                detected_defects[pole_number].append(defects)

            elif any(item_name.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):

                defects = self.search_defects(path_to_video=path_to_data)

                detected_defects[pole_number].append(defects)

            else:
                raise TypeError("ERROR: File's extension cannot be processed")

        elif os.path.isdir(path_to_data):

            for item in os.listdir(path_to_data):

                # TODO: We might not be able to process PNGs. Double check

                if any(item.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):

                    print("\nImage:", item)
                    path_to_image = os.path.join(path_to_data, item)

                    defects = self.search_defects(path_to_image=path_to_image)

                    # If successfully processed, store defects found
                    if defects:
                        detected_defects[pole_number].append(defects)
                    else:
                        detected_defects[pole_number].append({})

                elif any(item.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):

                    print("Video:", item)
                    path_to_video = os.path.join(path_to_data, item)

                    defects = self.search_defects(path_to_video=path_to_video)

                    # If successfully processed, store defects found
                    if defects:
                        detected_defects[pole_number].append(defects)
                    else:
                        detected_defects[pole_number].append({})
                else:
                    continue
        else:
            raise TypeError("ERROR: Wrong input. Neither folder nor file")

        return detected_defects

    def search_defects(
            self,
            path_to_image: str=None,
            path_to_video: str=None
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
            filename = os.path.splitext(os.path.basename(path_to_image))[0]

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
                video_writer = cv2.VideoWriter(os.path.join(self.save_path, 'video_out.avi'), fourcc, 5,
                                              (image_to_process.shape[1],
                                               image_to_process.shape[0]), True)

            # Process 1 in N frames to increase performance speed
            # TODO: Extrapolate BBs

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
                                                                       image_name=filename)
                defect_time = time.time() - start
                defects[filename] = detected_defects

            # Combine all objects detected into one dict for further processing
            detected_objects = {**poles, **components}

            self.handler.draw_bounding_boxes(objects_detected=detected_objects,
                                             image=image_to_process)

            self.handler.save_frame(image=image_to_process,
                                    image_name=filename,
                                    video_writer=video_writer)

            #cv2.imshow("Frame", image_to_process)

            frame_counter += 1

            print(f"Defects time: {round(defect_time, 3)}")

            fps.update()

        fps.stop()
        #print("Elapsed time: {:.2f}".format(fps.elapsed()))
        #print("Approx FPS: {:.2f}".format(fps.fps()))

        video_stream.stop()
        cv2.destroyAllWindows()

        return defects
