from app.visual_detector.workers import FrameReaderThread, ObjectDetectorThread
from app.visual_detector.workers import DefectDetectorThread, ResultsProcessorThread
from app.visual_detector.neural_networks import PolesDetector, ComponentsDetector
from app.visual_detector.defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from app.visual_detector.utils import ResultsHandler
import queue
import os
import uuid


class MainDetector:

    def __init__(
            self,
            save_path: str,
            db=None,
            search_defects: bool = True
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

        # Initialize detectors and auxiliary modules
        self.results_processor = ResultsHandler(save_path=save_path)
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
            components_detector=self.component_detector,
            progress=self.progress,
            batch_size=3
        )

        self.defect_detector_thread = DefectDetectorThread(
            in_queue=self.block1_to_block2,
            out_queue=self.block2_to_writer,
            defect_detector=self.defect_detector,
            progress=self.progress,
            check_defects=search_defects
        )

        self.results_processor_thread = ResultsProcessorThread(
            in_queue=self.block2_to_writer,
            save_path=save_path,
            results_processor=self.results_processor,
            progress=self.progress,
            database=db
        )

        # Launch threads and wait for a request
        self.start()

    def predict(
            self,
            request_id: str,
            path_to_data: str,
            pole_number: int
    ) -> dict:
        """
        API endpoint - parses input data, puts files provided to the Q if of appropriate extension
        :param path_to_data: Path to data to process - image, video, folder with images, video
        :return: dictionary {filename : ID, }
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
            print(f"\nERROR: file {filename} cannot be processed, wrong extension")
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
        :return:
        """
        # Each file to process gets a unique ID number to track its progress
        file_id = str(uuid.uuid4())

        # TODO: Store what needs to be done to a file: angle or defects calculations
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

    def start(self):
        for thread in (
            self.frame_reader_thread,
            self.object_detector_thread,
            self.defect_detector_thread,
            self.results_processor_thread
        ):
            thread.start()

    def stop(self):
        self.files_to_process_Q.put("STOP")
        # Wait until all threads complete their job and exit
        for thread in (
            self.frame_reader_thread,
            self.object_detector_thread,
            self.defect_detector_thread,
            self.results_processor_thread
        ):
            thread.join()
