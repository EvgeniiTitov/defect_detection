import threading
import cv2
import os
import numpy as np


class ResultsProcessorThread(threading.Thread):

    def __init__(
            self,
            save_path,
            queue_from_defect_detector,
            results_processor,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.progress = progress

        self.Q_in = queue_from_defect_detector
        self.results_processor = results_processor
        self.save_path = save_path
        self.previous_id = None
        self.video_writer = None

    def run(self) -> None:

        while True:
            input = self.Q_in.get()

            if input == "END":
                if self.previous_id:
                    self.save_video()

                continue

            elif input == "STOP":
                if self.previous_id:
                    self.save_video()

                break

            image, id, detected_objects = input
            pole_number = self.progress[id]['pole_id']
            filename = self.progress[id]['filename']

            store_path = os.path.join(self.save_path, str(pole_number))

            if id != self.previous_id:
                self.previous_id = id

                if not os.path.exists(store_path):
                    try:
                        os.mkdir(store_path)
                    except:
                        pass

            self.video_writer = None

            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(os.path.join(store_path, filename + '_out.avi'),
                                               fourcc, 30, (image.shape[1], image.shape[0]), True)

            # Draw BBs
            self.results_processor.draw_bounding_boxes(objects_detected=detected_objects,
                                                       image=image)

            # cv2.imshow("frame", image)
            # if cv2.waitKey(0):
            #     cv2.destroyAllWindows()

            self.video_writer.write(image.astype(np.uint8))

        print("ResultsProcessorThread killed")
        return

    def stop(self) -> None: self.done = True

    def save_video(self):
        self.video_writer = None
