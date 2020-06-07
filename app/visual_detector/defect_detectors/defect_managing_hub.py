from threading import Thread
from typing import Dict, List
from app.visual_detector.neural_networks import DetectedObject
from app.visual_detector.utils import TensorManager
import functools
import numpy as np
import torch
import os
import cv2
import time
import random


class DefectDetector:

    def __init__(
            self,
            line_modifier,
            concrete_extractor,
            concrete_cracks_detector=None,
            dumpers_defect_detector=None,
            insulators_defect_detector=None,
            wood_crack_detector=None,
            tilt_detector=None
    ):
        # Auxiliary modules
        self.concrete_extractor = concrete_extractor(line_modifier=line_modifier)

        # Defect detecting modules
        self.Q_to_dumper_classifier, self.Q_from_dumper_classifier = dumpers_defect_detector
        self.crack_classifier = concrete_cracks_detector
        self.insulator_classifier = insulators_defect_detector
        self.wood_crack_classifier = wood_crack_detector
        self.tilt_detector = tilt_detector

        # TODO: 1. How to organise code here so we can scale? Easily add new detectors, run some of them only,
        #          check which ones are running
        #       2. Do we want to be able to turn on / off some of them?

        print("Defect detector successfully initialized")

    def search_defects_on_frame(
            self,
            image_on_cpu: np.ndarray,
            image_on_gpu: torch.Tensor,
            towers: list,
            components: list
    ) -> dict:
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
        sliced_tensors = self.prepare_objs_for_defect_detection(detections=sorted_detections, image_on_gpu=image_on_gpu)

        # Each defect detector might require specific image processing, so it will be done directly
        # in the corresponding detector's thread
        # TODO: Valid idea?
        # Send sliced out tensors + objects they belong to corresponding detectors
        for class_name, elements in sliced_tensors.items():
            if class_name == "dumper":
                self.Q_to_dumper_classifier.put(elements)
            elif class_name == "pillar":
                pass
            elif class_name == "wood":
                pass
            else:
                pass

        # Get signals from threads that they have finished processing sliced tensors and have marked
        # corresponding DetectedObject instances as either defected or healthy
        # TODO: How to wait for threads to finish?

        for q in [self.Q_from_dumper_classifier]:
            q.get()

        return {}

    def prepare_objs_for_defect_detection(
            self,
            detections: Dict[str, List[DetectedObject]],
            image_on_gpu: torch.Tensor
    ) -> dict:
        """

        :param detections:
        :param image_on_gpu:
        :return:
        """
        output = dict()

        for class_name, elements in detections.items():

            output[class_name] = list()
            for element in elements:
                # Slightly widen pillar's bb to ensure proper mask application
                if class_name == "pillar":
                    TensorManager.modify_pillar_bb(pillar=element, image=image_on_gpu)

                # Get object's coordinates and slice out the tensor
                left = element.left
                top = element.top
                right = element.right
                bot = element.bottom
                element_bb_image = TensorManager.slice_out_tensor(
                    image=image_on_gpu,
                    coordinates=[left, top, right, bot]
                )
                # Index 0 - object, index 1 - corresponding sliced tensor
                output[class_name].append([element, element_bb_image])

        return output

    def sort_objects(self, towers: list, components: list) -> Dict[str, List[DetectedObject]]:
        """

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

    def search_defects(
            self,
            detected_objects: dict,
            image: np.ndarray,
            image_name: str = None,
            camera_orientation: tuple=None
    ) -> dict:
        """
        Finds defects on objects provided
        :param detected_objects: objects
        :param camera_orientation: metadata in case image and its got metadata
        :param pole_number: number of the pole which image's getting processed
        :return:
        """
        detected_defects = {
            "pillar": [],
            "dumper": [],
            "insulator": []
        }
        # Subimage is either a subimage of a pole if any have been detected or the whole original
        # image ib case no pole have been found. Elements are objects detected within this subimage
        for subimage, elements in detected_objects.items():
            for index, element in enumerate(elements):
                if element.object_name.lower() == "pillar":

                    # Does not return anything. Change object's state - declare it defected or not
                    self.search_defects_pillars(
                        pillar=element,
                        frame=image,
                        subimage=subimage,
                        camera_angle=camera_orientation,
                        image_name=image_name
                    )

                    # If inclination was successfully calculated
                    if element.inclination is not None:
                        detected_defects["pillar"].append(("angle", element.inclination))

                    if element.cracked:
                        pass

                elif element.object_name.lower() == "dump":

                    dumper_subimage = np.array(image[subimage.top + element.BB_top:subimage.top + element.BB_bottom,
                                                     subimage.left + element.BB_left:subimage.left + element.BB_right])

                    # print("Processing:", image_name)
                    # msg = [subimage.top + element.BB_top, subimage.top + element.BB_bottom,\
                    #        subimage.left + element.BB_left, subimage.left + element.BB_right]
                    # print(msg)
                    try:
                        cv2.imwrite(
                            os.path.join("D:\Desktop\system_output\OUTPUT\dumpers", str(random.randint(0,10**7)) + ".jpg"),
                            img=dumper_subimage
                        )
                    except Exception as e:
                        print(f"Failed to saved the cropped dumper. Error: {e}")
                        exit()

                    # Search for defects on vibration dumpers
                    continue

                elif element.object_name.lower() == "insul":
                    # Search for defects on insulators
                    continue

        return detected_defects

    def search_defects_pillars(
            self,
            pillar,
            frame,
            subimage,
            camera_angle,
            image_name
    ):
        """
        Method for detecting pillars inclination and cracks
        :param pillar:
        :param detection_section:
        :return:
        """
        # Reconstruct image of the pillar detected
        pillar_subimage = np.array(frame[pillar.top + subimage.top:pillar.bottom + subimage.top,
                                         pillar.left + subimage.left:pillar.right + subimage.left])

        # cv2.imwrite(
        #     os.path.join("D:\Desktop\system_output\API_RESULTS\cropped", image_name + '.jpg'),
        #     pillar_subimage)

        # Search for pole's edges
        func_to_time = timeout(seconds=10)(self.concrete_extractor.find_pole_edges)
        try:
            pillar_edges = func_to_time(image=pillar_subimage)
        except:
            print("Timeout error raised")
            return

        if not pillar_edges:
            return

        # for edge in pillar_edges:
        #     cv2.line(pillar_subimage, edge[0], edge[1], (0, 0, 255), 4)
        # cv2.imwrite(
        #     os.path.join("D:\Desktop\system_output\API_RESULTS\lines", image_name + '.jpg'), pillar_subimage)

        # Run inclination calculation
        inclination = self.calculate_angle(the_lines=pillar_edges)
        pillar.inclination = inclination

        # Run cracks detection
        # concrete_polygon = self.concrete_extractor.retrieve_polygon_v2(the_edges=pillar_edges,
        #                                                                image=pillar_subimage)

        # TODO: Send polygon for cracks detection


        # cv2.imwrite(
        #    os.path.join("D:\Desktop\system_output\RESULTS\cropped", image_name + '.jpg'), concrete_polygon)

        # cv2.imshow("cropped", concrete_polygon)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def calculate_angle(self, the_lines):
        """
        Calculates angle of the line(s) provided
        :param the_lines: list of lists, lines found and filtered
        :return: angle
        """
        if len(the_lines) == 2:
            x1_1 = the_lines[0][0][0]
            y1_1 = the_lines[0][0][1]
            x2_1 = the_lines[0][1][0]
            y2_1 = the_lines[0][1][1]
            angle_1 = round(90 - np.rad2deg(np.arctan2(abs(y2_1 - y1_1), abs(x2_1 - x1_1))), 2)

            x1_2 = the_lines[1][0][0]
            y1_2 = the_lines[1][0][1]
            x2_2 = the_lines[1][1][0]
            y2_2 = the_lines[1][1][1]
            angle_2 = round(90 - np.rad2deg(np.arctan2(abs(y2_2 - y1_2), abs(x2_2 - x1_2))), 2)

            the_angle = round((angle_1 + angle_2) / 2, 2)
            assert 0 <= the_angle <= 90, "ERROR: Wrong angle value calculated"

            return the_angle

        else:
            x1 = the_lines[0][0][0]
            y1 = the_lines[0][0][1]
            x2 = the_lines[0][1][0]
            y2 = the_lines[0][1][1]

            the_angle = round(90 - np.rad2deg(np.arctan2(abs(y2 - y1), abs(x2 - x1))), 2)
            assert 0 <= the_angle <= 90, "ERROR: Wrong angle value calculated"

            return the_angle


def timeout(seconds):
    # https://stackoverflow.com/questions/21827874/timeout-a-function-windows
    def deco(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception(f"Function {func.__name__} timeout {seconds} exceeded")]

            def new_func():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=new_func)
            t.daemon = True

            try:
                t.start()
                # Attempt joining the thread in N seconds
                t.join(timeout=seconds)
            except Exception as e:
                raise e

            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret

            return ret
        return wrapper
    return deco
