import threading
import numpy as np


"""
Inside defect detector you can have N number of threads each tasked with
finding defects on an object of particular class. Connected by Qs, return
results to one place (join threads), which get put in Q_out and sent for postprocessing

VS multiprocessing. Might be faster but data transfer overhead
"""


class DefectDetectorThread(threading.Thread):

    def __init__(
            self,
            queue_from_object_detector,
            queue_to_results_processor,
            defect_detector,
            defects,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_object_detector
        self.Q_out = queue_to_results_processor

        self.defect_detector = defect_detector
        self.defects = defects

    def run(self) -> None:

        while not self.done:

            # Get a dictionary of detected objects
            item = self.Q_in.get(block=True)

            if item == "END":
                self.Q_out.put("END")
                break

            image, poles, components = item

            if components:
                detected_defects = self.defect_detector.search_defects(components)

                # Add only if any defects have been found
                if any(detected_defects[key] for key in detected_defects.keys()):
                    self.defects["defects"].append(detected_defects)

            self.Q_out.put((image, {**poles, **components}))

        return
