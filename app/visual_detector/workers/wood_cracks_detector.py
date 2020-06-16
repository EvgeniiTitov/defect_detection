import threading
import random
import numpy as np
import cv2
import os


class WoodCracksDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            wood_tower_segment,
            wood_cracks_detector,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.wood_tower_segmentation = wood_tower_segment
        self.crack_detector = wood_cracks_detector
        self.progress = progress

    def run(self) -> None:
        while True:

            input_ = self.Q_in.get()
            if input_ == "END":
                break

            file_id, items = input_

            # Collect sliced out bb of wood towers
            images_to_preprocess = list()
            for obj, np_array in items:
                assert isinstance(np_array, np.ndarray), "Wood tower subimage is not np.ndarray"
                images_to_preprocess.append(np_array)

            # Extract masks of wood towers without background
            masks = self.wood_tower_segmentation.process_batch(images_to_preprocess)

            # Run crack detecting algorithm on the extracted masks
            results = list()
            for mask in masks:
                # TODO: IMPORTANT - consider slciing masks into N parts to improve
                #       crack detection algorithm. How to keep track of results?
                results.append(self.crack_detector.detect_cracks(mask))

            name = os.path.splitext(os.path.basename(self.progress[file_id]['path_to_file']))[0]
            for i, mask in enumerate(results):
                cv2.imwrite(
                    os.path.join(
                        "D:\Desktop\system_output\OUTPUT\segmentations", f"{name}_{i}_out.jpg"
                    ),
                    mask
                )

            self.Q_out.put("Success")

        print("WoodCracksDetectorThread successfully killed")
