import threading
import cv2
import os


class FrameReaderThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.progress = progress

    def run(self) -> None:

        while True:

            input_ = self.Q_in.get()

            # If input is STOP -> kill the thread
            if input_ == "STOP":
                break

            file_id = input_
            path_to_file = self.progress[file_id]["path_to_file"]

            # If failed to open a file, signal to other threads the current video is over
            try:
                cap = cv2.VideoCapture(path_to_file)
            except:
                print("\nERROR: Failed to open file:", os.path.basename(path_to_file))
                self.Q_out.put("END")
                continue

            if not cap.isOpened():
                self.Q_out.put("END")
                continue

            # Save total number of frames to process
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress[file_id]["total_frames"] = total_frames
            self.progress[file_id]["status"] = "Being processed"

            while True:
                # For an image reads it ones and breaks out
                has_frame, frame = cap.read()

                if not has_frame:
                    break

                self.progress[file_id]["now_processing"] += 1
                self.Q_out.put((frame, file_id))

            cap.release()
            self.Q_out.put("END")

        self.Q_out.put("STOP")
        print("FrameReaderThread killed")
