import threading


class DefectTrackingThread(threading.Thread):
    '''
    Tracks objects from frame to frame - to be implemented
    Generates JSON, saves results to DB when required
    '''
    def __init__(
            self,
            in_queue,
            out_queue,
            object_tracker,
            results_saver,
            progress,
            database,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.Q_in = in_queue
        self.Q_out = out_queue
        self.object_tracker = object_tracker
        self.results_saver = results_saver
        self.progress = progress
        self.database = database

        self.previous_id = None

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
                batch_frames, file_id, detections, index_checked_frame = input_
            except Exception as e:
                print(f"Failed to unpack the message from DefectDetectorThread. Error: {e}")
                raise e

            '''
            TO BE IMPLEMENTED:
            1. Objects need to be tracked. Now each frame is considered as a separate image, as a result the 
               defect detection results cannot be linked between different frames 
            '''

            # If any components have been detected, a frame with the most number of elements on it selected and sent
            # for defect detection, iterate over components on this frame saving their deficiency status
            defects_on_frame = dict()
            if index_checked_frame:
                components = detections[index_checked_frame]
                for i, component in enumerate(components):

                    if component.object_name not in defects_on_frame.keys():
                        defects_on_frame[component.object_name] = list()

                    if component.inclination and component.object_name == "pillar":
                        defects_on_frame[component.object_name].append(
                            (f"index_{i}", "angle", component.inclination)
                        )
                    if component.object_name == "dumper":
                        defects_on_frame[component.object_name].append(
                            (f"index_{i}", "missing_weight", component.is_weight_missing)
                        )
                    if component.object_name == "wood" and component.cracked:
                        defects_on_frame[component.object_name].append(
                            (f"index_{i}", "cracked_wood", component.cracked)
                        )

            # TODO: Drop results to the dictionary and add it to self.progress[file_id]["defects"]
            #       You can drop from here, but trigger JSON generation from result processor ! Good idea actually

            # TODO: Wood integration.
            # TODO: Tests

            # print('\n\n')
            # for img_batch_index, elements in detections.items():
            #     for element in elements:
            #         print(f"Img index: {img_batch_index}. Element name: {element.object_name}. "
            #               f"Deficiency status: {element.deficiency_status}. Angle: {element.inclination}."
            #               f" Lines: {element.edges}")

                    # TODO FIX RESULT PROCESSOR: Draw line, run 1k images + video test. Test speed


            # TODO: Loop over detected objects in the batch and generate an entry to the dictionary that later will
            # be written to JSON. Include only entries from the frame that was sent for defect detection.
            # So you need this frame from the defect detector

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

            self.Q_out.put((batch_frames, file_id, detections))

        print("DefectTrackingThread killed")
