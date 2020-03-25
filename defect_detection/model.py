from concurrency import FrameReaderThread, ObjectDetectorThread
from concurrency import DefectDetectorThread, ResultsProcessorThread
from neural_networks import PolesDetector, ComponentsDetector
from defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from utils import ResultsHandler
from collections import defaultdict
import queue
import os
import time
import cv2
import uuid


# TODO: Method to stop the system where you join threads
# TODO: How to return results to the user? Asyncio vs ?


class MainDetector:

    def __init__(
            self,
            save_path: str,
            search_defects: bool = True
    ):
        # Path on the server where processed data gets stored
        self.save_path = save_path
        self.search_defects = search_defects

        # To keep track of video processing (how many frames processed)
        # TODO: When to clean it? If server doesn't get restarted in a while
        self.progress = dict()

        if search_defects:
            self.check_defects = True
            # Metadata extractor could be initialized here if required
        else:
            self.check_defects = False

        self.results_processor = ResultsHandler(save_path=save_path)

        # Initialize detectors
        self.pole_detector = PolesDetector()
        self.component_detector = ComponentsDetector()
        self.defect_detector = DefectDetector(
            line_modifier=LineModifier,
            concrete_extractor=ConcreteExtractor,
            cracks_detector=None,
            dumpers_defect_detector=None,
            insulators_defect_detector=None
        )

        # Initialize Qs and workers
        self.files_to_process_Q = queue.Queue()
        self.frame_to_block1 = queue.Queue(maxsize=24)
        self.block1_to_block2 = queue.Queue(maxsize=6)
        self.block2_to_writer = queue.Queue(maxsize=10)

        self.frame_reader_thread = FrameReaderThread(
            in_queue=self.files_to_process_Q,
            out_queue=self.frame_to_block1,
            progress=self.progress
        )

        self.object_detector_thread = ObjectDetectorThread(
            in_queue=self.frame_to_block1,
            out_queue=self.block1_to_block2,
            poles_detector=self.pole_detector,
            components_detector=self.component_detector
        )

        self.defect_detector_thread = DefectDetectorThread(
            in_queue=self.block1_to_block2,
            out_queue=self.block2_to_writer,
            defect_detector=self.defect_detector,
            check_defects=search_defects
        )

        self.results_processor_thread = ResultsProcessorThread(
            in_queue=self.block2_to_writer,
            save_path=save_path,
            results_processor=self.results_processor,
            progress=self.progress
        )

        #self.start()

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
            defects = self.process_file(path_to_file=path_to_data,
                                        pole_number=pole_number)

            if defects is not None:
                detected_defects[pole_number].append(defects)

        elif os.path.isdir(path_to_data):

            for item in os.listdir(path_to_data):
                path_to_file = os.path.join(path_to_data, item)
                defects = self.process_file(path_to_file=path_to_file,
                                            pole_number=pole_number)

                if defects is not None:
                    detected_defects[pole_number].append(defects)
        else:
            print("ERROR: Cannot process the file:", path_to_data)

        return detected_defects

    def process_file(
            self,
            path_to_file: str,
            pole_number: int
    ):
        """
        Checks file's extension, calls appropriate method
        :return:
        """
        filename = os.path.basename(path_to_file)

        if any(filename.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):
            print("\nProcessing image:", filename)
            return self.process_image(path_to_image=path_to_file,
                                      pole_number=pole_number)

        elif any(filename.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):
            print("\nProcessing video:", filename)
            return self.process_video(path_to_video=path_to_file,
                                      pole_number=pole_number)
        else:
            print(f"\nERROR: Ext {os.path.splitext(filename)[-1]} cannot be processed")
            return None

    def process_image(
            self,
            path_to_image: str,
            pole_number: int
    ) -> dict:
        """
        TODO: Could check for metadata if required
        :param path_to_image:
        :param pole_number:
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
    ):
        """
        :param path_to_video:
        :param pole_number:
        :return:
        """
        # Defects from each frame will be stored there
        detected_defects = defaultdict(list)

        # Each video to process gets a unique ID number to track its progress
        video_id = str(uuid.uuid4())

        # Keep track of processing progress
        self.progress[video_id] = {
            "pole_number": pole_number,
            "path_to_video": path_to_video,
            "processing": 0,
            "processed": 0,
            "remaining": None
        }

        self.files_to_process_Q.put((path_to_video, pole_number, video_id))

    def start(self):
        for thread in (
            self.frame_reader_thread,
            self.object_detector_thread,
            self.defect_detector_thread,
            self.results_processor_thread
        ):
            thread.start()

    def stop(self):
        for thread in (
            self.frame_reader_thread,
            self.object_detector_thread,
            self.defect_detector_thread,
            self.results_processor_thread
        ):
            thread.join()
