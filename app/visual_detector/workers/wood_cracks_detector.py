import threading


class WoodCracksDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            wood_cracks_detector,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.model = wood_cracks_detector
        self.progress = progress

    def run(self) -> None:
        while True:
            input = self.Q_in.get()
            print("Input:", input)
