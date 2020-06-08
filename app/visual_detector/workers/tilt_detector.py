import threading


class TiltDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            tilt_detector,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.detector = tilt_detector
        self.progress = progress

    def run(self) -> None:
        while True:

            input_ = self.Q_in.get()
            if input_ == "STOP":
                break

            # Collect sliced out numpy arrays to get processed
            pillars_to_process = list()
            for obj, pillar in input_:
                pillars_to_process.append(pillar)

            # Get results from the detector
            output = self.detector.find_pole_edges_calculate_angle(pillars=pillars_to_process)
            assert len(output) == len(pillars_to_process), "Nb of pillars and results do not match"

            for i in range(len(output)):
                angle = output[i][0]
                if angle:
                    input_[i][0].inclination = angle
                    input_[i][0].edges.append(output[i][1])

            self.Q_out.put("Success")

        print("TiltDetectorThread successfully killed")
