import threading
import cv2
import os
import numpy as np


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
        self.in_q = in_queue
        self.out_q = out_queue
        self.progress = progress

    def run(self) -> None:

        while True:
            input = self.in_q.get()

            if input == 'STOP':
                break

            (path_to_data, pole_id, id) = input

            try:
                stream = cv2.VideoCapture(path_to_data)
            except:
                print("Failed to open the file", os.path.basename(path_to_data))
                self.out_q.put('END')
                continue

            if not stream.isOpened():
                self.out_q.put('END')
                continue

            # TODO: Remaining frames count (set 'remaining')

            while True:
                has_frame, frame = stream.read()
                if not has_frame:
                    self.out_q.put('END')
                    break

                # Blocks the thread till there's a place in the Q to put an item
                self.progress[id]['processing'] += 1  # We only need approximate progress, non-atomic ops are ok
                self.out_q.put((frame, id))

            stream.release()
            self.out_q.put('END')

        self.out_q.put('STOP')
        print("\nFrameReaderThread killed")

    def get_frame(self) -> np.ndarray: return self.Q.get()

    def has_frame(self) -> bool: return self.Q.qsize() > 0
