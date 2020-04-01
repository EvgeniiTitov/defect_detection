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

    def run(self) -> None:

        while True:

            input_ = self.Q_in.get()

            if input_ == "STOP":
                self.Q_out.put("STOP")
                break

            if input_ == "END":
                self.Q_out.put("END")
                continue

            frame, poles, components, file_id = input_

            #TODO: Check what frame's is getting processed - run inference ones in N frames

            if components and self.check_defects:
                detected_defects = self.defect_detector.search_defects(components)

                #TODO: Any found defects -> save to the progress file under file's ID
                #      for video it might mean adding defects for each frame ->

                # Add only if any defects have been found - do not add info about frames
                # where no defects have been detected
                if any(detected_defects[key] for key in detected_defects.keys()):

                    # TODO: CHECK - NOW ADDING RESULTS TO THE LIST (list of dicts - ugly)
                    self.progress[file_id]["defects"].append(detected_defects)

            self.Q_out.put((frame, file_id, {**poles, **components}))

        print("DefectDetectorThread killed")
