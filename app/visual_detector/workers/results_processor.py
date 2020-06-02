from datetime import datetime
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
            database,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.results_processor = results_processor
        self.save_path = save_path
        self.progress = progress
        self.database = database

        self.video_writer = None
        self.previous_id = None

    def run(self) -> None:

        while True:
            input_ = self.Q_in.get()

            # Check if video is over or it is time to kill the workers
            if input_ == "STOP":
                break
            if input_ == "END":
                if self.previous_id:
                    self.clean_video_writer()  # Video's over. Delete current video writer
                continue

            # Read data from defect detector
            try:
                (batch_frames, file_id, detected_objects) = input_
            except Exception as e:
                print(f"Failed to unpack a message from DefectDetectorThread. Error: {e}")
                raise e

            try:
                file_type = self.progress[file_id]["file_type"]
                pole_number = self.progress[file_id]["pole_number"]
                filename = os.path.splitext(os.path.basename(self.progress[file_id]["path_to_file"]))[0]
                store_path = os.path.join(self.save_path, str(pole_number))
            except Exception as e:
                print(f"Failed while reading file data. Error: {e}")
                raise e

            # New file arrived
            if file_id != self.previous_id:
                self.previous_id = file_id

            # TODO: Consider creating a flag for this so we do not check it for each video frame losing time
            if not os.path.exists(store_path):
                try:
                    os.mkdir(store_path)
                except Exception as e:
                    print(f"Failed to create a folder to save processed images. Error: {e}")
                    raise e

            if self.video_writer is None and file_type == "video":
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.video_writer = cv2.VideoWriter(
                    os.path.join(store_path, filename + "_out.avi"),
                    fourcc, 30, (batch_frames[0].shape[1], batch_frames[0].shape[0]), True
                )

            self.results_processor.draw_bb_on_batch(
                images=batch_frames,
                detections=detected_objects
            )

            if file_type == "video":
                self.results_processor.save_batch_on_disk(
                    images=batch_frames,
                    video_writter=self.video_writer
                )
            else:
                cv2.imwrite(
                    filename=os.path.join(store_path, filename + "_out.jpg"),
                    img=batch_frames[0]
                )
            self.progress[file_id]["processed"] += len(batch_frames)

            # ---DELETE ME---
            for frame in batch_frames:
                cv2.imshow("", frame)
                if cv2.waitKey(1):
                    break

            # After all frames processed, save results
            if self.progress[file_id]["processed"] >= self.progress[file_id]["total_frames"]:
                #self.save_results_to_db(file_id, filename, store_path)
                self.results_processor.save_results_to_json(
                    filename=filename,
                    store_path=store_path,
                    payload=self.progress[file_id]["defects"]
                )
                self.progress[file_id]["status"] = "Processed"
                del self.progress[file_id]
                print(f"Processing of {filename} completed")

        print("ResultsProcessorThread killed")

    def clean_video_writer(self):
        self.video_writer = None

    def save_results_to_db(self, file_id: int, filename: str, store_path: str) -> bool:
        """
        Dumps processing results into the database
        :param file_id:
        :param filename:
        :param store_path:
        :return:
        """
        try:
            # Creates a collection on the fly or finds the existing one
            prediction_results = self.database.db.predictions
            prediction_results.insert(
                {
                    "request_id": self.progress[file_id]["request_id"],
                    "file_id": file_id,
                    "file_name": filename,
                    "saved_to": store_path,
                    "datetime": datetime.utcnow(),
                    "defects": self.progress[file_id]["defects"]
                }
            )
            return True
        except Exception as e:
            print(f"Error while inserting into db: {e}")
            return False
