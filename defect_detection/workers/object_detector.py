import threading
#from pynvml import *

class ObjectDetectorThread(threading.Thread):

    def __init__(
            self,
            in_queue,
            out_queue,
            poles_detector,
            components_detector,
            progress,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.poles_detector = poles_detector
        self.components_detector = components_detector
        self.progress = progress

        # try:
        #     nvmlInit()
        # except Exception as e:
        #     print("Exception during pynvml initialization:", e)
        # self.device = nvmlDeviceGetHandleByIndex(0)

    def run(self) -> None:

        while True:

            # Blocks the thread if the Q's empty
            input_ = self.Q_in.get(block=True)

            # Check if its time to kill the thread
            if input_ == "STOP":
                self.Q_out.put("STOP")
                break

            # Check if the video is over or an image (1 frame) was processed
            if input_ == "END":
                self.Q_out.put("END")
                continue

            (frame, file_id) = input_
            self.progress[file_id]["now_processing"] += 1

            # usage = nvmlDeviceGetMemoryInfo(self.device)
            # usage_rate = nvmlDeviceGetUtilizationRates(self.device)
            # print(f"\nGPU: {usage_rate.gpu}, GPU-memory: {usage_rate.memory}%")
            # print("Total memory:", usage.total)
            # print("Free memory:", usage.free)
            # print("Used memory:", usage.used)

            # Predict poles, returns dict with (image: predicted poles)
            poles = self.poles_detector.predict(image=frame)

            # TODO: Check what to do to the file: inclination or defects
            #       predict all components OR just concrete poles. You could
            #       check object's information to see what needs to be done to it

            # Predict components, returns dict (pole_bb_subimage: components found within)
            components = self.components_detector.predict(image=frame,
                                                          pole_predictions=poles)

            self.Q_out.put((frame, poles, components, file_id))

        print("ObjectDetectorThread killed")
