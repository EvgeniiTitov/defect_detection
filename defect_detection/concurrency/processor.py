from concurrency.frame_reader import FrameReader
from concurrency.object_detector import ObjectDetector
from concurrency.defect_detector import DefectDetector
from concurrency.results_processor import ResultsProcessor
import queue
import os


"""
1. Am i correct and we will have to reinitialize all Qs and threads for each new file we want 
to process?

2. Should I have a separate method to process images without threads? Might be taking some time
to create all of those Qs and threads for one image. And for each image, we reinitialize them.
BUT, then you'll have to bring and initialize all your detectors and networks - dont want to do
it in the constructor, then will have to reinitialise for each photo! 

3. Can I use the same already initialized nets and other objects in both threads and for predicting
on single images? We just need to give reference to the threads

3. There's data that is revealed in one thread and is required in another (frame sizes for video
writer AND the frame itself, matrix) 
Should I be carrying the frame along the whole process?

4. Should I import YOLO, detectors and defect detector related classes here and provide them to the
threads during initialization or directly over there?


5. How to time out function
____
Need to add try - except blocks, so that if anything fails, all other threads die as well



"""


class Processor:

    def __init__(
            self,
            save_path: str,
            search_defects: bool=True
    ):
        self.save_path = save_path
        self.search_defects = search_defects

    def predict(
            self,
            path_to_data: str
    ) -> dict:
        """
        API endpoint method. Parses input
        :param path_to_data: Path to data to process - image, video, folder with images, video
        :return: dictionary {filename : defects, }
        """
        detected_defects = dict()

        if os.path.isfile(path_to_data):
            filename = os.path.basename(path_to_data)

            if any(filename.endswith(ext) for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]):

                # this is an image
                pass

            elif any(filename.endswith(ext) for ext in ["avi", "AVI", "MP4", "mp4"]):
                # this is a video
                pass

            else:
                raise TypeError(f"ERROR: Files of {os.path.splitext(filename)[-1]} "
                                f"extension cannot be processed")

        elif os.path.isdir(path_to_data):
            pass

        else:
            raise TypeError("ERROR: Provided file is neither a folder nor a file")

        return detected_defects

    def process_file(
            self,
            path_to_image: str=None,
            path_to_video: str=None
    ):
        """
        TBA
        :param path_to_image:
        :param path_to_video:
        :return:
        """
        frame_to_block1 = queue.Queue(maxsize=24)
        block1_to_block2 = queue.Queue(maxsize=4)
        block2_to_writer = queue.Queue(maxsize=10)

        file_to_process = path_to_image, "image" if path_to_image else path_to_video, "video"
        filename = os.path.splitext(os.path.basename(file_to_process[0]))

        frame_reader = FrameReader(path_to_data=file_to_process[0],
                                   queue=frame_to_block1)

        object_detector = ObjectDetector(queue_from_frame_reader=frame_to_block1,
                                         queue_to_defect_detector=block1_to_block2)

        defect_detector = DefectDetector(queue_from_object_detector=block1_to_block2,
                                         queue_to_results_processor=block2_to_writer)

        result_processor = ResultsProcessor(save_path=self.save_path,
                                            queue_from_defect_detector=block2_to_writer,
                                            input_type=file_to_process[1],
                                            filename=filename)

        # Start all threads
        frame_reader.start()
        object_detector.start()
        defect_detector.start()
        result_processor.start()

        # Wait until all threads complete
        frame_reader.join()
        object_detector.join()
        defect_detector.join()
        result_processor.join()



if __name__ == "__main__":

    path = r"D:\Desktop\Reserve_NNs\Datasets\raw_data\new_wooden_poles\MorePoles\filename.txt"
    print(os.path.splitext(os.path.basename(path))[0])