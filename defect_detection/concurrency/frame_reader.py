import threading
import cv2
import os
import numpy as np


class FrameReaderThread(threading.Thread):

    def __init__(
            self,
            path_to_data,
            queue,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q = queue
        self.done = False

        try:
            self.stream = cv2.VideoCapture(path_to_data)
        except:
            print("Failed to open the file", os.path.basename(path_to_data))
            self.Q.put("END")

    def run(self) -> None:
        while True:
            if self.done:
                break

            has_frame, frame = self.stream.read()
            if not has_frame:
                self.stop()
                break

            # Blocks the thread till there's a place in the Q to put an item
            self.Q.put(frame)

        self.stream.release()
        self.Q.put("END")

        print("\nFrameReaderThread killed")
        return

    def get_frame(self) -> np.ndarray: return self.Q.get()

    def has_frame(self) -> bool: return self.Q.qsize() > 0

    def stop(self) -> None: self.done = True
