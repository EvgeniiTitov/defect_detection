from app.visual_detector.workers import FrameReaderThread, PoleDetectorThread, ComponentDetectorThread
from app.visual_detector.workers import DefectDetectorThread, ResultsProcessorThread
from app.visual_detector.neural_networks import TowerDetector, ComponentsDetector
from app.visual_detector.defect_detectors import DefectDetector, LineModifier, ConcreteExtractor
from app.visual_detector.utils import ResultsHandler
import queue
import os
import uuid


class MainDetector:

    def __init__(
            self,
            save_path: str,
            batch_size: int,
            search_defects: bool = True,
            db=None,
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
        self.pole_detector = TowerDetector()
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
        self.frame_to_pole_detector = queue.Queue(maxsize=24)
        self.pole_to_comp_detector = queue.Queue(maxsize=24)
        self.comp_to_defect_detector = queue.Queue(maxsize=6)
        self.defect_to_writer = queue.Queue(maxsize=10)

        self.frame_reader_thread = FrameReaderThread(
            batch_size=batch_size,
            in_queue=self.files_to_process_Q,
            out_queue=self.frame_to_pole_detector,
            progress=self.progress
        )

        self.pole_detector_thread = PoleDetectorThread(
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
            out_queue=self.defect_to_writer,
            defect_detector=self.defect_detector,
            progress=self.progress,
            check_defects=search_defects
        )

        self.results_processor_thread = ResultsProcessorThread(
            in_queue=self.defect_to_writer,
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
            self.results_processor_thread
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
            self.results_processor_thread
        ):
            thread.join()
