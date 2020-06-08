import threading
import cv2
import os


class ResultsProcessorThread(threading.Thread):

    def __init__(
            self,
            save_path,
            in_queue,
            drawer,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.drawer = drawer
        self.save_path = save_path
        self.progress = progress

        self.video_writer = None
        self.previous_id = None
        self.folder_created = False

    def run(self) -> None:

        while True:
            input_ = self.Q_in.get()

            # Check if video is over or it is time to kill the workers
            if input_ == "STOP":
                break
            if input_ == "END":
                if self.previous_id:
                    self.clean_video_writer()  # Video's over. Delete current video writer
                    self.folder_created = False
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

            # Check if the folder to which results will be saved exists
            if not self.folder_created and not os.path.exists(store_path):
                try:
                    os.mkdir(store_path)
                    self.folder_created = True
                except Exception as e:
                    print(f"Failed to create a folder to save processed images. Error: {e}")
                    raise e

            # Create video writer if required
            if self.video_writer is None and file_type == "video":
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.video_writer = cv2.VideoWriter(
                    os.path.join(store_path, filename + "_out.avi"),
                    fourcc, 30, (batch_frames[0].shape[1], batch_frames[0].shape[0]), True
                )

            # Draw bounding boxes and save image on disk
            self.drawer.draw_bb_on_batch(
                images=batch_frames,
                detections=detected_objects
            )
            if file_type == "video":
                self.drawer.save_batch_on_disk(
                    images=batch_frames,
                    video_writter=self.video_writer
                )
            else:
                self.drawer.save_image_on_disk(
                    save_path=store_path,
                    image_name=filename + ".jpg",
                    image=batch_frames[0]
                )

            self.progress[file_id]["processed"] += len(batch_frames)

            #---DELETE ME---
            for frame in batch_frames:
                cv2.imshow("", frame)
                if cv2.waitKey(1):
                    break

            # After all frames processed, save results
            if self.progress[file_id]["processed"] >= self.progress[file_id]["total_frames"]:
                #self.save_results_to_db(file_id, filename, store_path)
                # self.results_processor.save_results_to_json(
                #     filename=filename,
                #     store_path=store_path,
                #     payload=self.progress[file_id]["defects"]
                # )
                self.progress[file_id]["status"] = "Processed"
                del self.progress[file_id]
                print(f"Processing of {filename} completed")

        print("ResultsProcessorThread killed")

    def clean_video_writer(self):
        self.video_writer = None
