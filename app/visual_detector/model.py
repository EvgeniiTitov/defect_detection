from app.visual_detector.workers import (
    FrameReaderThread, TowerDetectorThread, ComponentDetectorThread, DefectDetectorThread,
    DefectTrackingThread, ResultsProcessorThread, TiltDetectorThread, DumperClassifierThread,
    WoodCracksDetectorThread
)
from app.visual_detector.neural_networks import (
    TowerDetector, ComponentsDetector, DumperClassifier, WoodCrackSegmenter
)
from app.visual_detector.defect_detectors import DefectDetector, LineModifier, ConcretePoleHandler
from app.visual_detector.utils import Drawer, ResultSaver, ObjectTracker
import queue
import os
import uuid
import time


class MainDetector:

    def __init__(
            self,
            save_path: str,
            batch_size: int,
            search_defects: bool = True,
            db=None
    ):
        # Path on the server where processed data gets stored
        self.save_path = save_path
        self.search_defects = search_defects
        # To keep track of file processing
        self.progress = dict()

        if search_defects:
            self.check_defects = True
            # Metadata extractor could be initialized here if required
        else:
            self.check_defects = False

        # Initialize Qs and workers
        try:
            # --- video / image processors ---
            self.files_to_process_Q = queue.Queue()
            self.frame_to_pole_detector = queue.Queue(maxsize=3)
            self.pole_to_comp_detector = queue.Queue(maxsize=3)
            self.comp_to_defect_detector = queue.Queue(maxsize=3)
            self.defect_det_to_defect_tracker = queue.Queue(maxsize=3)
            self.defect_tracker_to_writer = queue.Queue(maxsize=3)

            # --- dumper classifier ---
            self.to_dumper_classifier = queue.Queue(maxsize=3)
            self.from_dumper_classifier = queue.Queue(maxsize=3)
            # --- tilt detector ---
            self.to_tilt_detector = queue.Queue(maxsize=3)
            self.from_tilt_detector = queue.Queue(maxsize=3)
            # --- wood tower cracks ---
            self.to_wood_cracks_detector = queue.Queue(maxsize=3)
            self.from_wood_cracks_detector = queue.Queue(maxsize=3)
            # --- conrete tower cracks ---
            pass
        except Exception as e:
            print(f"Failed during queue initialization. Error: {e}")
            raise e

        # Initialize detectors and auxiliary modules
        try:
            # Object detectors
            self.pole_detector = TowerDetector()
            self.component_detector = ComponentsDetector()

            # Defect detectors
            self.defect_detector = DefectDetector(
                tilt_detector=(self.to_tilt_detector, self.from_tilt_detector),
                dumpers_defect_detector=(self.to_dumper_classifier, self.from_dumper_classifier),
                wood_crack_detector=(self.to_wood_cracks_detector, self.from_wood_cracks_detector),
                concrete_cracks_detector=None,
                insulators_defect_detector=None
            )
            self.wood_tower_segment = WoodCrackSegmenter()
            self.dumper_classifier = DumperClassifier()

            # Auxiliary modules
            self.drawer = Drawer(save_path=save_path)
            self.result_saver = ResultSaver()
            self.object_tracker = ObjectTracker()

        except Exception as e:
            print(f"Failed during detectors initialization. Error: {e}")
            raise e

        try:
            self.frame_reader_thread = FrameReaderThread(
                batch_size=batch_size,
                in_queue=self.files_to_process_Q,
                out_queue=self.frame_to_pole_detector,
                progress=self.progress
            )
            self.pole_detector_thread = TowerDetectorThread(
                in_queue=self.frame_to_pole_detector,
                out_queue=self.pole_to_comp_detector,
                poles_detector=self.pole_detector,
                progress=self.progress
            )
            self.component_detector_thread = ComponentDetectorThread(
                in_queue=self.pole_to_comp_detector,
                out_queue=self.comp_to_defect_detector,
                component_detector=self.component_detector,
                progress=self.progress
            )
            self.defect_detector_thread = DefectDetectorThread(
                in_queue=self.comp_to_defect_detector,
                out_queue=self.defect_det_to_defect_tracker,
                defect_detector=self.defect_detector,
                progress=self.progress,
                check_defects=search_defects
            )
            self.defect_tracker = DefectTrackingThread(
                in_queue=self.defect_det_to_defect_tracker,
                out_queue=self.defect_tracker_to_writer,
                object_tracker=self.object_tracker,
                results_saver=self.result_saver,
                progress=self.progress,
                database=db
            )
            self.results_processor_thread = ResultsProcessorThread(
                in_queue=self.defect_tracker_to_writer,
                save_path=save_path,
                drawer=self.drawer,
                progress=self.progress
            )

            # ------ DEFECT DETECTORS ------
            self.dumper_classifier_thread = DumperClassifierThread(
                in_queue=self.to_dumper_classifier,
                out_queue=self.from_dumper_classifier,
                dumper_classifier=self.dumper_classifier,
                progress=self.progress
            )
            self.tilt_detector_thread = TiltDetectorThread(
                in_queue=self.to_tilt_detector,
                out_queue=self.from_tilt_detector,
                tilt_detector=ConcretePoleHandler(line_modifier=LineModifier),
                progress=self.progress
            )
            self.wood_cracks_thread = WoodCracksDetectorThread(
                in_queue=self.to_wood_cracks_detector,
                out_queue=self.from_wood_cracks_detector,
                wood_cracks_detector=None,
                wood_tower_segment=self.wood_tower_segment,
                progress=self.progress
            )
        except Exception as e:
            print(f"Failed during workers initialization. Error: {e}")
            raise e

        self.start()

    def predict(
            self,
            request_id: str,
            path_to_data: str,
            pole_number: int
    ) -> dict:
        """
        API endpoint - parses input data, puts files provided to the Q if of appropriate extension
        :param request_id:
        :param path_to_data:
        :param pole_number:
        :return:
        """
        file_IDS = {}

        if os.path.isfile(path_to_data):
            file_id = self.check_type_and_process(
                path_to_file=path_to_data,
                pole_number=pole_number,
                request_id=request_id
            )
            file_IDS[path_to_data] = file_id
            return file_IDS

        elif os.path.isdir(path_to_data):
            for item in os.listdir(path_to_data):
                path_to_file = os.path.join(path_to_data, item)
                file_id = self.check_type_and_process(
                    path_to_file=path_to_file,
                    pole_number=pole_number,
                    request_id=request_id
                )
                file_IDS[path_to_file] = file_id
        else:
            print(f"ERROR: Cannot process the file {path_to_data}, neither a folder nor a file")

        return file_IDS

    def check_type_and_process(
            self,
            path_to_file: str,
            pole_number: int,
            request_id: str
    ) -> str:
        """
        Checks whether a file can be processed
        :return: file's ID if it was put in the Q or "None" if extension is not supported
        """
        filename = os.path.basename(path_to_file)

        if any(filename.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):
            print(f"Added image {filename} to the processing queue")
            return self.process_file(
                path_to_file=path_to_file,
                pole_number=pole_number,
                file_type="image",
                request_id=request_id
            )

        elif any(filename.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):
            print(f"Added video {filename} to the processing queue")
            return self.process_file(
                path_to_file=path_to_file,
                pole_number=pole_number,
                file_type="video",
                request_id=request_id
            )
        else:
            print(f"ERROR: file {filename}'s extension is not supported")
            return "None"

    def process_file(
            self,
            path_to_file: str,
            pole_number: int,
            file_type: str,
            request_id: str
    ) -> str:
        """

        :param path_to_file:
        :param pole_number:
        :param file_type:
        :param request_id:
        :return:
        """
        # Each file to process gets a unique ID number to track its progress
        file_id = str(uuid.uuid4())

        # Keep track of processing progress
        self.progress[file_id] = {
            "status": "Awaiting processing",
            "request_id": request_id,
            "file_type": file_type,
            "pole_number": pole_number,
            "path_to_file": path_to_file,
            "total_frames": None,
            "now_processing": 0,
            "processed": 0,
            "defects": []
        }
        self.files_to_process_Q.put(file_id)

        return file_id

    def start(self) -> None:
        for thread in (
            self.frame_reader_thread,
            self.pole_detector_thread,
            self.component_detector_thread,
            self.defect_detector_thread,
            self.defect_tracker,
            self.results_processor_thread,
            self.dumper_classifier_thread,
            self.tilt_detector_thread,
            self.wood_cracks_thread
        ):
            thread.start()

    def stop(self) -> None:
        self.files_to_process_Q.put("STOP")
        # Wait until all threads complete their job and exit
        for thread in (
            self.frame_reader_thread,
            self.pole_detector_thread,
            self.component_detector_thread,
            self.defect_detector_thread,
            self.defect_tracker,
            self.results_processor_thread,
            self.dumper_classifier_thread,
            self.tilt_detector_thread,
            self.wood_cracks_thread
        ):
            thread.join()
