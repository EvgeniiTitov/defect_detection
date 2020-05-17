import threading


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
        self.poles_detector = poles_detector
        self.progress = progress
        self.video_id = None

    def run(self) -> None:
        while True:
            # Blocks the thread if the Q's empty
            input_ = self.Q_in.get(block=True)

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

            # Predict poles, returns dict with (image: predicted poles)
            poles = self.poles_detector.predict_batch(images=gpu_batch_frames)
            #poles = self.poles_detector.predict(batch_frames[0])

            self.Q_out.put((batch_frames, gpu_batch_frames, file_id, poles))

        print("PoleDetectorThread killed")
