import threading
import cv2
import os
import numpy as np
import json


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
                break

            if input_ == "END":
                if self.previous_id:
                    self.clean_video_writer()  # Video's over. Delete current video writer
                continue

            (frame, file_id, detected_objects) = input_

            file_type = self.progress[file_id]["file_type"]
            pole_number = self.progress[file_id]["pole_number"]
            filename = os.path.splitext(os.path.basename(self.progress[file_id]["path_to_file"]))[0]
            store_path = os.path.join(self.save_path, str(pole_number))

            # New file arrived
            if file_id != self.previous_id:
                self.previous_id = file_id

            if not os.path.exists(store_path):
                try:
                    os.mkdir(store_path)
                except Exception as e:
                    print(f"Failed to create a folder to save processed images. Error: {e}")
                    pass

            if self.video_writer is None and file_type == "video":
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.video_writer = cv2.VideoWriter(os.path.join(store_path, filename + "_out.avi"),
                                                    fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            self.results_processor.draw_bounding_boxes(
                objects_detected=detected_objects,
                image=frame
            )
            if file_type == "video":
                self.video_writer.write(frame.astype(np.uint8))
            else:
                cv2.imwrite(
                    filename=os.path.join(store_path, filename + "_out.jpg"),
                    img=frame
                )

            self.progress[file_id]["processed"] += 1

            if self.progress[file_id]["processed"] == self.progress[file_id]["total_frames"]:

                # TODO: Dump results to Postgres

                with open(os.path.join(store_path, filename + ".json"), "w") as f:
                    json.dump(self.progress[file_id]["defects"], f)
                self.progress[file_id]["status"] = "Processed"

                # TODO: Once dumped, clean the progress tracking dictionary
                #del self.progress[file_id]

                print(f"Processing of {filename} completed")

        print("ResultsProcessorThread killed")

    def clean_video_writer(self):
        self.video_writer = None
