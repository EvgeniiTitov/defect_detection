import threading


class ObjectDetectorThread(threading.Thread):

    def __init__(
            self,
            queue_from_frame_reader,
            queue_to_defect_detector,
            poles_detector,
            components_detector,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_frame_reader
        self.Q_out = queue_to_defect_detector

        self.poles_detector = poles_detector
        self.components_detector = components_detector

    def run(self) -> None:

        while not self.done:

            # Block the thread if the Q's empty
            input = self.Q_in.get(block=True)
            print("OBJ DETECTOR: Got a frame from Q - predicting objects")

            # Check if its time to kill the thread
            if input == "END":
                self.Q_out.put(input)
                continue

            elif input == "STOP":
                self.Q_out.put(input)
                break

            frame, id = input

            # Predict poles, returns dict with (whole frame: predicted poles)
            poles = self.poles_detector.predict(image=frame)

            # Predict components, returns dict (pole_subimage: components found)
            components = self.components_detector.predict(image=frame,
                                                          pole_predictions=poles)

            print("OBJ DETECTOR: Put predicted objects in the Q")
            # Put results in one dict and send forward
            self.Q_out.put((frame, poles, components, id))

        print("ObjectDetectorThread killed")
        return

    def stop(self) -> None: self.done = True
