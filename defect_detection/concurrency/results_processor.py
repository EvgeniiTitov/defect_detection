import threading
import cv2
import os
import numpy as np


class ResultsProcessorThread(threading.Thread):

    def __init__(
            self,
            save_path,
            queue_from_defect_detector,
            filename,
            pole_number,
            results_processor,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_defect_detector
        self.filename = filename
        self.results_processor = results_processor

        self.store_path = os.path.join(save_path, str(pole_number))
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)

    def run(self) -> None:

        video_writer = None
        while not self.done:

            item = self.Q_in.get(block=True)

            if item == "END":
                break

            image, detected_objects = item

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(os.path.join(self.store_path, self.filename + '_out.avi'),
                                               fourcc, 30,(image.shape[1], image.shape[0]), True)

            # Draw BBs
            self.results_processor.draw_bounding_boxes(objects_detected=detected_objects,
                                                       image=image)

            video_writer.write(image.astype(np.uint8))

        return
