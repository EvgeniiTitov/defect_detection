from threading import Thread, Event
import numpy as np
import os
import cv2
import time


class DefectDetector:
    """
    TO DO: Consider multiprocessing
    """
    def __init__(
            self,
            line_modifier,
            concrete_extractor,
            cracks_detector=None,
            dumpers_defect_detector=None,
            insulators_defect_detector=None,
    ):

        # Auxiliary modules
        self.line_modifier = line_modifier
        self.concrete_extractor = concrete_extractor(line_modifier=self.line_modifier)

        # Defect detecting modules
        self.cracks_tester = cracks_detector
        self.dumper_tester = dumpers_defect_detector
        self.insulator_tester = insulators_defect_detector

        # Memorize the edges once they have been generated
        self.the_lines = None

        print("Defect detecting hub initialized\n")

    def search_defects(
            self,
            detected_objects: dict,
            camera_orientation: tuple,
            pole_number: int,
            image_name: str
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

                    # Do not return anything. Change object's state - declare it defected or not
                    self.pillars_defects_detector(pillar=element,
                                                  detection_section=subimage,
                                                  camera_angle=camera_orientation,
                                                  image_name=image_name)


                    detected_defects["pillar"].append(("angle", element.inclination))

                    if element.cracked:
                        pass

                elif element.object_name.lower() == "dump":
                    # Search for defects on vibration dumpers
                    continue

                elif element.object_name.lower() == "insul":
                    # Search for defects on insulators
                    continue

        return detected_defects

    def pillars_defects_detector(
            self,
            pillar,
            detection_section,
            camera_angle,
            image_name
    ):
        """
        Pipeline for detecting pillars inclination and cracks
        :param pillar:
        :param detection_section:
        :return:
        """
        # Generate lines
        # Reconstruct image of the pillar detected
        pillar_subimage = np.array(detection_section.frame[pillar.top:pillar.bottom,
                                                           pillar.left:pillar.right])

        # cv2.imwrite(
        #     os.path.join("D:\Desktop\system_output\RESULTS\cropped_to_test_separately", image_name + '.jpg'),
        #     pillar_subimage)

        # Search for pole's edges
        pillar_edges = self.concrete_extractor.find_pole_edges(image=pillar_subimage)

        if not pillar_edges:
            return

        for edge in pillar_edges:
            cv2.line(pillar_subimage, edge[0], edge[1], (0, 0, 255), 4)

        cv2.imwrite(
            os.path.join("D:\Desktop\system_output\RESULTS\lines", image_name + '.jpg'), pillar_subimage)

        # Run inclination calculation
        inclination = self.calculate_angle(the_lines=pillar_edges)

        if inclination and 0 <= inclination <= 90:
            pillar.inclination = inclination
        else:
            pillar.inclination = "NULL"

        # Run cracks detection
        concrete_polygon = self.concrete_extractor.retrieve_polygon_v2(the_edges=pillar_edges,
                                                                       image=pillar_subimage)

        # TO DO: Send polygon for cracks detection


        cv2.imwrite(
           os.path.join("D:\Desktop\system_output\RESULTS\cropped", image_name + '.jpg'), concrete_polygon)

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

            return round((angle_1 + angle_2) / 2, 2)

        else:
            x1 = the_lines[0][0][0]
            y1 = the_lines[0][0][1]
            x2 = the_lines[0][1][0]
            y2 = the_lines[0][1][1]

            return round(90 - np.rad2deg(np.arctan2(abs(y2 - y1), abs(x2 - x1))), 2)
