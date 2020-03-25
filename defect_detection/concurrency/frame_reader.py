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
            # Check if there's a video to process
            input_ = self.Q_in.get()

            # If input is STOP -> kill the thread
            if input_ == "STOP":
                break

            (path_to_video, pole_number, video_id) = input_

            # If failed to open a video, signal to other threads the current video is over
            # by sending END
            try:
                stream = cv2.VideoCapture(path_to_video)
            except:
                print("\nERROR: Failed to open video:", os.path.basename(path_to_video))
                self.Q_out.put("END")
                continue

            if not stream.isOpened():
                self.Q_out.put("END")
                continue

            # Save total number of frames to process
            total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress[video_id]["remaining"] = total_frames

            while True:
                has_frame, frame = stream.read()

                if not has_frame:
                    break

                self.progress[video_id]["processing"] += 1
                self.Q_out.put((frame, video_id))

            stream.release()
            self.Q_out.put("END")

        self.Q_out.put("STOP")
        print("FrameReaderThread killed")
