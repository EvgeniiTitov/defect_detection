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

            '''
            2. После сегментационной нейронки, режим столбы на части. 
            3. Находим трещины. 
            '''

            # Collect sliced out bb of wood towers
            images_to_preprocess = list()
            for obj, np_array in input_:
                assert isinstance(np_array, np.ndarray), "Wood tower subimage is not np.ndarray"
                images_to_preprocess.append(np_array)

                # cv2.imwrite(
                #     os.path.join(
                #         r"D:\Desktop\system_output\Wood_Crack_Test",
                #         f"{str(random.randint(0, 1000))}_out.jpg"
                #     ),
                #     np_array
                # )

            masks = self.wood_tower_segmentation.process_batch(images_to_preprocess)

            for i, mask in enumerate(masks):
                cv2.imwrite(
                    os.path.join("D:\Desktop\system_output\OUTPUT", f"{i}_out.jpg"),
                    mask
                )

            self.Q_out.put("Success")

        print("WoodCracksDetectorThread successfully killed")
