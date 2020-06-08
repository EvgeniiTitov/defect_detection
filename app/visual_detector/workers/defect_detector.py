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
            check_defects,
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

            try:
                batch_frame, gpu_batch_frame, file_id, towers, components = input_
            except Exception as e:
                print(f"Failed to unpack a message from the ComponentDetectorThread. Error: {e}")
                raise e
            '''
            We need to run self.defect_detector only if we have found objects for which we have running detectors 
            
            Run defect detector on 1 frame in the batch with the most objects found on it 
            
            1. Check if any components have been detected 
            2. If any, send components and images on gpu for defect detection
            3. Makes sense to handle any detected defects here, result processor just draws boxes, saves results
            4. Combine towers and components in one dictionary 
            '''

            # 1. Check if any objects've been detected and number of objects on each frame.
            # 2. Collect all detections in one dictionary to send it to the results processor
            detections_summary = {i: set() for i in range(len(batch_frame))}
            detections_overall = 0
            combined_detections = defaultdict(list)
            for d in [towers, components]:
                for img_batch_index, detections in d.items():
                    combined_detections[img_batch_index].extend(detections)
                    for detection in detections:
                        detections_summary[img_batch_index].add(detection.object_name)
                        detections_overall += 1

            # TODO: Check if we have any detectors running for any detected objects - create a list of all
            #       running detectors in main model.py and give it to the thread similar to self.progress
            # If any objects found, check on which frame the most nb of objects detected and send it for defect
            # detection
            index_frame_check_defects = None
            if detections_overall > 0 and self.check_defects:
                most_detections = 0
                index_frame_check_defects = 0
                for i, detections in detections_summary.items():
                    if len(detections) > most_detections:
                        most_detections = len(detections)
                        index_frame_check_defects = i

                # Does not return anything - modifies DetectedObject instances state to declare them defected
                self.defect_detector.search_defects_on_frame(
                    image_on_cpu=batch_frame[index_frame_check_defects],
                    image_on_gpu=gpu_batch_frame[index_frame_check_defects],
                    towers=towers[index_frame_check_defects],
                    components=components[index_frame_check_defects]
                )
            del gpu_batch_frame

            self.Q_out.put((batch_frame, file_id, combined_detections, index_frame_check_defects))

        print("DefectDetectorThread killed")
