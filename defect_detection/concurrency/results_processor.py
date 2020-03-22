import threading
import cv2
import os
import numpy as np


class ResultsProcessorThread(threading.Thread):

    def __init__(
            self,
            save_path,
            queue_from_defect_detector,
            filename,
            pole_number,
            results_processor,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_defect_detector
        self.filename = filename
        self.results_processor = results_processor
        self.save_path = save_path
        self.pole_number = pole_number

    def run(self) -> None:

        store_path = os.path.join(self.save_path, str(self.pole_number))
        if not os.path.exists(store_path):
            try:
                os.mkdir(store_path)
            except:
                print(f"Failed to create a folder to store results for {self.pole_number} pole")
                self.stop()
                # TODO: How do I stop other threads from here?

        video_writer = None
        while not self.done:

            item = self.Q_in.get(block=True)
            print("POSTPROCESSOR: Get predicted objects - postprocessing")

            if item == "END":
                break

            image, detected_objects = item

            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(os.path.join(store_path, self.filename + '_out.avi'),
                                               fourcc, 30,(image.shape[1], image.shape[0]), True)

            # Draw BBs
            self.results_processor.draw_bounding_boxes(objects_detected=detected_objects,
                                                       image=image)

            cv2.imshow("frame", image)
            if cv2.waitKey(0):
                cv2.destroyAllWindows()

            video_writer.write(image.astype(np.uint8))

        print("ResultsProcessorThread killed")
        return

    def stop(self) -> None: self.done = True
