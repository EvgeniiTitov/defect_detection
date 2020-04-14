import threading
import numpy as np


"""
Inside defect detector you can have N number of threads each tasked with
finding defects on an object of particular class. Connected by Qs, return
results to one place (join threads), which get put in Q_out and sent for postprocessing
FOR loop across all elements found putting them in appropriate Qs

VS multiprocessing. Might be faster but data transfer overhead - apache arrow 
"""


class DefectDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            defect_detector,
            progress,
            check_defects=True,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.defect_detector = defect_detector
        self.check_defects = check_defects
        self.progress = progress
        self.currently_processing = 0

    def run(self) -> None:

        while True:

            input_ = self.Q_in.get()

            if input_ == "STOP":
                self.Q_out.put("STOP")
                break

            if input_ == "END":
                self.Q_out.put("END")
                self.refresh_counter()
                continue

            frame, poles, components, file_id = input_

            if components and self.check_defects and self.currently_processing % 10 == 0:
                detected_defects = self.defect_detector.search_defects(components)

                # Add only if any defects have been found - do not add info about frames
                # where no defects have been detected
                if any(detected_defects[key] for key in detected_defects.keys()):
                    self.progress[file_id]["defects"].append(detected_defects)

            self.currently_processing += 1
            self.Q_out.put((frame, file_id, {**poles, **components}))

        print("DefectDetectorThread killed")

    def refresh_counter(self):
        self.currently_processing = 0
