from concurrency.frame_reader import FrameReaderThread
from concurrency.object_detector import ObjectDetectorThread
from concurrency.defect_detector import DefectDetectorThread
from concurrency.results_processor import ResultsProcessorThread
from neural_networks import PolesDetector, ComponentsDetector
from neural_networks import YOLOv3
from defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from utils import ResultsHandler
from collections import defaultdict
import queue
import os
import time
import cv2


"""
Need to add try - except blocks, so that if anything fails, all other threads die as well
"""

class MainDetectorV2:

    def __init__(
            self,
            save_path: str,
            search_defects: bool = True
    ):
        self.save_path = save_path
        self.search_defects = search_defects

        if search_defects:
            self.check_defects = True
            # Metadata extractor can be initialized here
        else:
            self.check_defects = False

        poles_network = YOLOv3()
        components_network = YOLOv3()
        pillars_network = YOLOv3()

        self.pole_detector = PolesDetector(poles_network)
        self.component_detector = ComponentsDetector(components_predictor=components_network,
                                                     pillar_predictor=pillars_network)

        self.defect_detector = DefectDetector(line_modifier=LineModifier,
                                              concrete_extractor=ConcreteExtractor,
                                              cracks_detector=None,
                                              dumpers_defect_detector=None,
                                              insulators_defect_detector=None)

        self.results_processor = ResultsHandler(save_path=save_path)

    def predict(
            self,
            path_to_data: str,
            pole_number: int
    ) -> dict:
        """
        API endpoint method.
        :param path_to_data: Path to data to process - image, video, folder with images, video
        :return: dictionary {filename : defects, }
        """
        detected_defects = defaultdict(list)

        if os.path.isfile(path_to_data):
            filename = os.path.basename(path_to_data)

            if any(filename.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):
                print("\nProcessing image:", filename)
                defects = self.process_image(path_to_image=path_to_data,
                                             pole_number=pole_number)

                detected_defects[pole_number].append(defects)

            elif any(filename.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):
                print("\nProcessing video:", filename)
                defects = self.process_video(path_to_video=path_to_data,
                                             pole_number=pole_number)

                detected_defects[pole_number].append(defects)

            else:
                print(f"ERROR: Ext {os.path.splitext(filename)[-1]} cannot be processed")
                return {}

        elif os.path.isdir(path_to_data):
            for item in os.listdir(path_to_data):

                if any(item.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):
                    print("\nProcessing image:", item)
                    path_to_image = os.path.join(path_to_data, item)
                    defects = self.process_image(path_to_image=path_to_image,
                                                 pole_number=pole_number)

                    detected_defects[pole_number].append(defects)

                elif any(item.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):
                    print("\nProcessing video:", item)
                    path_to_video = os.path.join(path_to_data, item)
                    defects = self.process_video(path_to_video=path_to_video,
                                                 pole_number=pole_number)

                    detected_defects[pole_number].append(defects)

                else:
                    print("Cannot process:", item)
        else:
            print("Cannot process the file:", path_to_data)
            return {}

        return detected_defects

    def process_image(
            self,
            path_to_image: str,
            pole_number: int
    ) -> dict:
        """
        TODO: Could check for metadata if required
        :param path_to_image:
        :return:
        """
        try:
            image = cv2.imread(filename=path_to_image)
        except:
            print("Failed to open:", os.path.basename(path_to_image))
            return {}

        # Discard ext, get just image's name
        image_name = os.path.splitext(os.path.basename(path_to_image))[0]

        t1 = time.time()
        poles = self.pole_detector.predict(image=image)
        components = self.component_detector.predict(image=image,
                                                     pole_predictions=poles)
        obj_detection = time.time() - t1


        defect_detection = 0
        detected_defects = {image_name : {}}
        if components and self.check_defects:
            t2 = time.time()
            detected_defects[image_name] = self.defect_detector.search_defects(detected_objects=components)
            defect_detection = time.time() - t2

        self.results_processor.draw_bb_save_image(image=image,
                                                  detected_objects={**poles, **components},
                                                  pole_number=pole_number,
                                                  image_name=image_name)

        print("Time taken: "
              f"Object detection {round(obj_detection, 3)} "
              f"Defect detection {round(defect_detection, 3)}")

        return detected_defects

    def process_video(
            self,
            path_to_video: str,
            pole_number: int
    ) -> list:
        """
        TBA
        :param path_to_video:
        :return:
        """
        # Defects from each frame will be stored there
        detected_defects = defaultdict(list)

        frame_to_block1 = queue.Queue(maxsize=24)
        block1_to_block2 = queue.Queue(maxsize=6)
        block2_to_writer = queue.Queue(maxsize=10)
        filename = os.path.splitext(os.path.basename(path_to_video))[0]

        frame_reader = FrameReaderThread(path_to_data=path_to_video,
                                         queue=frame_to_block1)

        object_detector = ObjectDetectorThread(queue_from_frame_reader=frame_to_block1,
                                               queue_to_defect_detector=block1_to_block2,
                                               poles_detector=self.pole_detector,
                                               components_detector=self.component_detector)

        defect_detector = DefectDetectorThread(queue_from_object_detector=block1_to_block2,
                                               queue_to_results_processor=block2_to_writer,
                                               defect_detector=self.defect_detector,
                                               defects=detected_defects)

        result_processor = ResultsProcessorThread(save_path=self.save_path,
                                                  queue_from_defect_detector=block2_to_writer,
                                                  filename=filename,
                                                  pole_number=pole_number,
                                                  results_processor=self.results_processor)

        for thread in (frame_reader, object_detector, defect_detector, result_processor):
            thread.start()

        for thread in (frame_reader, object_detector, defect_detector, result_processor):
            thread.join()

        # Check if any defects have been found
        if "defects" in detected_defects.keys():
            return detected_defects["defects"]
        else:
            return []
