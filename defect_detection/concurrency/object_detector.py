from neural_networks import YOLOv3
from neural_networks import PolesDetector, ComponentsDetector
import threading


class ObjectDetector(threading.Thread):

    def __init__(
            self,
            queue_from_frame_reader,
            queue_to_defect_detector,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_frame_reader
        self.Q_out = queue_to_defect_detector

        poles_net = YOLOv3()
        components_net = YOLOv3()
        pillars_net = YOLOv3()

        self.pole_detector = PolesDetector(detector=poles_net)
        self.component_detector = ComponentsDetector(components_predictor=components_net,
                                                     pillar_predictor=pillars_net)

    def run(self) -> None:

        while not self.done:

            # Block the thread if the Q's empty, cannot make predictions
            # for None object - frame
            frame = self.Q_in.get(block=True)

            # Check if its time to kill the thread
            if frame == "END":
                self.Q_out.put("END")
                break

            # Predict poles, returns dict with (whole frame: predicted poles)
            poles = self.pole_detector.predict(image=frame)

            # Predict components, returns dict (pole_subimage: components found)
            components = self.component_detector.predict(image=frame,
                                                         pole_predictions=poles)

            # Put results in one dict and send forward
            self.Q_out.put((frame, {**poles, **components}))

        return

    def stop(self) -> None:

        self.done = True
