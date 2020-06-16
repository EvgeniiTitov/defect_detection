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

            try:
                batch_frames, gpu_batch_frames, towers, file_id = input_
            except Exception as e:
                print(f"Failed to unpack values from the input Q. Error: {e}")
                raise e

            # Check if any pole's been detected happens in the component detector
            components = self.component_detector.process_batch(
                images_on_gpu=gpu_batch_frames,
                towers_predictions=towers
            )
            self.Q_out.put((batch_frames, gpu_batch_frames, file_id, towers, components))

        print("ComponentDetectorThread killed")
