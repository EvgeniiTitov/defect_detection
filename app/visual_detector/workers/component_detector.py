import threading


class ComponentDetectorThread(threading.Thread):
    def __init__(
            self,
            in_queue,
            out_queue,
            component_detector,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.component_detector = component_detector
        self.progress = progress

    def run(self) -> None:
        while True:
            input_ = self.Q_in.get()
            if input_ == "STOP":
                self.Q_out.put("STOP")
                break
            elif input_ == "END":
                self.Q_out.put("END")  # video's over
                continue

            batch_frames, gpu_batch_frames, file_id, poles = input_
            # Check if any pole's been detected happens in the component detector
            components = self.component_detector.predict_batch(
                images=gpu_batch_frames,
                poles=poles
            )

            self.Q_out.put((batch_frames, gpu_batch_frames, file_id, poles, components))

        print("ComponentDetectorThread killed")
