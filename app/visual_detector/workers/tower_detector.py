import threading
from app.visual_detector.utils import ResultProcessor


class PoleDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            poles_detector,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.tower_detector = poles_detector
        self.progress = progress

    def run(self) -> None:
        while True:
            # Blocks the thread if the Q's empty
            input_ = self.Q_in.get()

            # Check if its time to kill the thread
            if input_ == "STOP":
                self.Q_out.put("STOP")
                break
            # Check if the video is over or an image (1 frame) was processed
            if input_ == "END":
                self.Q_out.put("END")
                continue

            (batch_frames, gpu_batch_frames, file_id) = input_
            self.progress[file_id]["now_processing"] += len(batch_frames)

            # Detect power line towers
            towers = self.tower_detector.process_batch(images_on_gpu=gpu_batch_frames)
            self.Q_out.put((batch_frames, gpu_batch_frames, towers, file_id))

        print("PoleDetectorThread killed")
