import threading
import cv2
import os
import numpy as np


class ResultsProcessorThread(threading.Thread):

    def __init__(
            self,
            save_path,
            in_queue,
            results_processor,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.results_processor = results_processor
        self.save_path = save_path
        self.progress = progress

        self.video_writer = None
        self.previous_id = None

    def run(self) -> None:

        while True:

            input_ = self.Q_in.get()

            if input_ == "STOP":
                if self.previous_id:
                    self.save_video()
                break

            if input_ == "END":
                if self.previous_id:
                    self.save_video()
                continue

            (frame, video_id, defects, detected_objects) = input_

            pole_number = self.progress[video_id]["pole_number"]
            filename = os.path.basename(self.progress[video_id]["path_to_video"])
            store_path = os.path.join(self.save_path, pole_number)

            if self.previous_id != video_id:
                self.previous_id = video_id

            if not os.path.exists(store_path):
                try:
                    os.mkdir(store_path)
                except:
                    print("ERROR: Failed to create a folder to save:", filename)
                    pass

            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.video_writer = cv2.VideoWriter(os.path.join(store_path, filename + "_out.avi"),
                                                    fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            self.results_processor.draw_bounding_boxes(objects_detected=detected_objects,
                                                       image=frame)

            self.video_writer.write(frame.astype(np.uint8))

        print("ResultsProcessorThread killed")

    def save_video(self):
        self.video_writer = None
