import threading


class ObjectDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            poles_detector,
            components_detector,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.poles_detector = poles_detector
        self.components_detector = components_detector

    def run(self) -> None:

        while True:

            # Blocks the thread if the Q's empty
            input_ = self.Q_in.get(block=True)

            # Check if its time to kill the thread
            if input_ == "STOP":
                self.Q_out.put("STOP")
                break

            # Check if the video is over
            if input_ == "END":
                self.Q_out.put("END")
                continue

            (frame, video_id) = input_

            # Predict poles, returns dict with (image: predicted poles)
            poles = self.poles_detector.predict(image=frame)
            # Predict components, returns dict (pole_bb: components found)
            components = self.components_detector.predict(image=frame,
                                                          pole_predictions=poles)

            # Put results in one dict and send forward
            self.Q_out.put((frame, poles, components, video_id))

        print("ObjectDetectorThread killed")
        return
