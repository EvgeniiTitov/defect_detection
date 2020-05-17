import threading
import cv2
import os
from app.visual_detector.utils import HostDeviceManager


class FrameReaderThread(threading.Thread):
    def __init__(
            self,
            batch_size,
            in_queue,
            out_queue,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.progress = progress

    def run(self) -> None:
        while True:
            input_ = self.Q_in.get()
            # If input == STOP -> kill the thread
            if input_ == "STOP":
                break

            file_id = input_
            path_to_file = self.progress[file_id]["path_to_file"]
            file_type = self.progress[file_id]["file_type"]

            # If failed to open a file, signal to other threads the current video is over
            try:
                cap = cv2.VideoCapture(path_to_file)
            except Exception as e:
                print(f"\nERROR: Failed to open file: {os.path.basename(path_to_file)}. Error: {e}")
                self.Q_out.put("END")
                continue

            if not cap.isOpened():
                self.Q_out.put("END")
                continue

            # Save total number of frames to process
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress[file_id]["total_frames"] = total_frames
            self.progress[file_id]["status"] = "Being processed"

            batch_frames = list()
            to_break = False
            while True:
                if len(batch_frames) < self.batch_size and file_type == "video":
                    has_frame, frame = cap.read()
                    if not has_frame:
                        to_break = True
                    else:
                        batch_frames.append(frame)
                        continue
                elif file_type == "image":
                    has_frame, frame = cap.read()
                    if not has_frame:
                        to_break = True
                    else:
                        batch_frames.append(frame)

                if batch_frames:
                    # Move batch of frames to GPU
                    try:
                        gpu_batch_frames = HostDeviceManager.load_images_to_GPU(batch_frames)
                    except Exception as e:
                        print(f"Failed to move a batch of frames to GPU. Error: {e}")
                        raise

                    # Send original images, imaged and GPU and file id to the next worker
                    print(f"Sending batch of size {len(gpu_batch_frames)} to pole detector")
                    self.Q_out.put((batch_frames, gpu_batch_frames, file_id))

                if to_break:
                    batch_frames = list()
                    break

                batch_frames = list()

            cap.release()
            self.Q_out.put("END")

        self.Q_out.put("STOP")
        print("FrameReaderThread killed")
