from typing import Dict, List
from .object_detector import DetectedObject
from app.visual_detector.utils import TensorManager
import numpy as np
import torch
import random
import cv2
import os


class DefectDetector:

    def __init__(
            self,
            running_detectors: List[str],
            tilt_detector=None,
            concrete_cracks_detector=None,
            dumpers_defect_detector=None,
            insulators_defect_detector=None,
            wood_crack_detector=None
    ):
        self.running_detectors = running_detectors
        self.Q_to_tilt_detector, self.Q_from_tilt_detector = tilt_detector
        self.Q_to_dumper_classifier, self.Q_from_dumper_classifier = dumpers_defect_detector
        self.Q_to_wood_cracks_detector, self.Q_from_wood_cracks_detector = wood_crack_detector

        self.concrete_crack_classifier = concrete_cracks_detector
        self.insulator_classifier = insulators_defect_detector

        # To kill threads if signalled so
        self.Qs_to = [
            self.Q_to_wood_cracks_detector,
            self.Q_to_tilt_detector,
            self.Q_to_dumper_classifier
        ]

        print("Defect detector successfully initialized")

    def kill_defect_detecting_threads(self):
        for q in self.Qs_to:
            q.put("STOP")

    def search_defects_on_frame(
            self,
            image_on_cpu: np.ndarray,
            image_on_gpu: torch.Tensor,
            towers: list,
            components: list,
            file_id: int
    ) -> None:
        """

        :param image_on_cpu:
        :param image_on_gpu:
        :param towers:
        :param components:
        :return:
        """
        '''
        Each detection is represented as the DetectedObject instance which has a number of defects related attributes 
        such as the .deficiency_status. The aim of this module is to process the detections and explicitly declare 
        detections defected should they have been classified as so. 
        
        1. Traverse over all detected objects and sort them into classes
        2. Do necessary image preprocessing
        4. Give batches of components of different classes to corresponding defect detectors 
        5. Get results, process them by declaring the corresponding detections as defected 
        '''
        # Sort objects by their class
        sorted_detections = self.sort_objects(towers, components)

        # Slice out tensors for each class remembering to what object each sliced out tensor belongs
        sliced_tensors = self.prepare_objs_for_defect_detection(
            detections=sorted_detections,
            image_on_gpu=image_on_gpu,
            image_on_cpu=image_on_cpu
        )
        # Each defect detector might require specific image processing, so it will be done directly
        # in the corresponding detector's thread
        # Send sliced out tensors + objects they belong to corresponding detector
        threads_to_wait = list()
        for class_name, elements in sliced_tensors.items():
            if class_name == "dumper" and "dumper" in self.running_detectors:
                self.Q_to_dumper_classifier.put(elements)
                threads_to_wait.append(self.Q_from_dumper_classifier)

            elif class_name == "pillar" and "pillar" in self.running_detectors:
                self.Q_to_tilt_detector.put(elements)
                threads_to_wait.append(self.Q_from_tilt_detector)

            elif class_name == "wood" and "wood" in self.running_detectors:
                self.Q_to_wood_cracks_detector.put((file_id, elements))
                threads_to_wait.append(self.Q_from_wood_cracks_detector)

            else:
                pass

        # Get signals from threads that they have finished processing and have marked
        # corresponding DetectedObject instances as either defected or healthy
        if threads_to_wait:
            for q in threads_to_wait:
                q.get()

        return

    def prepare_objs_for_defect_detection(
            self,
            detections: Dict[str, List[DetectedObject]],
            image_on_gpu: torch.Tensor,
            image_on_cpu: np.ndarray
    ) -> Dict[str, list]:
        """

        :param detections:
        :param image_on_gpu:
        :param image_on_cpu:
        :return:
        """
        output = dict()
        for class_name, elements in detections.items():
            if not class_name in output.keys():
                output[class_name] = list()

            # Do not have any processors for these classes yet
            if class_name == "concrete":
                continue
            elif class_name == "insulator":
                continue
            elif class_name == "metal":
                continue

            for element in elements:
                # Slightly widen pillar's bb to ensure proper mask application. Image on cpu and gpu dims are equal
                if class_name == "pillar":
                    TensorManager.modify_pillar_bb(pillar=element, image=image_on_gpu)

                # Get object's coordinates and slice out the tensor
                top = element.BB_top
                bot = element.BB_bottom
                # Get modified bb coordinates in the pillar's case
                if class_name == "pillar":
                    left = element.left
                    right = element.right
                else:
                    left = element.BB_left
                    right = element.BB_right

                # Tilt detecting algorithm is done on CPU, so slice out numpy array instead
                if class_name in ["pillar", "wood"]:
                    element_bb_image = TensorManager.slice_out_np_array(
                        image=image_on_cpu,
                        coordinates=[left, top, right, bot]
                    )

                    # save_path = r"D:\Desktop\system_output\wood_bb\from_spark"
                    # name = f"{random.randint(0, 10**5)}_out.jpg"
                    # try:
                    #     cv2.imwrite(os.path.join(save_path, name), element_bb_image)
                    # except Exception as e:
                    #     print(f"Failed while saving image. Error: {e}")
                    #     pass

                else:
                    element_bb_image = TensorManager.slice_out_tensor(
                        image=image_on_gpu,
                        coordinates=[left, top, right, bot]
                    )
                # Index 0 - object, index 1 - corresponding sliced tensor / numpy array
                output[class_name].append([element, element_bb_image])

        return output

    def sort_objects(self, towers: list, components: list) -> Dict[str, List[DetectedObject]]:
        """
        Receives lists of detected towers and components and sorts them by their class.
        :param towers:
        :param components:
        :return:
        """
        output = dict()
        for l in (towers, components):
            for element in l:
                object_class = element.object_name

                if object_class not in output.keys():
                    output[object_class] = list()

                output[object_class].append(element)

        return output
