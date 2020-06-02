import threading
from collections import defaultdict


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

            try:
                batch_frame, gpu_batch_frame, file_id, towers, components = input_
            except Exception as e:
                print(f"Failed to unpack a message from the ComponentDetectorThread. Error: {e}")
                raise e
            '''
            1. Check if any components have been detected 
            2. If any, send components and images on gpu for defect detection
            3. Makes sense to handle any detected defects here, result processor just draws boxes, saves results
            4. Combine towers and components in one dictionary 
            '''

            # if components and self.check_defects and self.currently_processing % 10 == 0:
            #     detected_defects = self.defect_detector.search_defects(
            #         detected_objects=components,
            #         image=frame,
            #         image_name=self.progress[file_id]["path_to_file"]
            #     )
            #
            #     # Add only if any defects have been detected
            #     if any(detected_defects[key] for key in detected_defects.keys()):
            #         self.progress[file_id]["defects"].append(detected_defects)

            del gpu_batch_frame

            self.currently_processing += len(batch_frame)
            # Merge dictionaries into one
            assert list(towers.keys()).sort() == list(components.keys()).sort(), "batch index keys do not match"
            output = defaultdict(list)
            for d in (towers, components):
                for key, value in d.items():
                    output[key].append(value)

            self.Q_out.put((batch_frame, file_id, output))

        print("DefectDetectorThread killed")

    def refresh_counter(self):
        self.currently_processing = 0
