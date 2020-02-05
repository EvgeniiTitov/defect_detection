from .concrete_extractor import ConcreteExtractor
from .line_modifier import LineModifier
import numpy as np
import sys
import cv2

#How to make it scalabable? It needs to be given *tools to use*. Like give it a detector for each defect


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

        # Cache the lines once they have been found
        self.the_lines = None

        print("Defect detecting hub initialized")

    def search_defects(
            self,
            detected_objects,
            camera_orientation,
            pole_number
    ):
        """
        Finds defects on objects provided
        :param detected_objects: objects
        :param camera_orientation: metadata in case image and its got metadata
        :param pole_number: number of the pole which image's getting processed
        :return:
        """
        # Subimage is either a subimage of a pole if any have been detected or the whole original
        # image ib case no pole have been found. Elements are objects detected within this subimage

        for subimage, elements in detected_objects.items():

            for element in elements:
                if element.object_name.lower() == "pillar":

                    # Do not return anything. Change object's state - declare it defected or not
                    self.pillars_defects_detector(pillar=element,
                                                  detection_section=subimage,
                                                  camera_angle=camera_orientation)

                elif element.object_name.lower() == "dump":
                    # Search for defects on vibration dumpers
                    continue

                elif element.object_name.lower() == "insul":
                    # Search for defects on insulators
                    continue

    def pillars_defects_detector(
            self,
            pillar,
            detection_section,
            camera_angle
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

        # Search for pole's edges
        pillar_edges = self.concrete_extractor.find_pole_edges(image=pillar_subimage)

        if not pillar_edges:
            return

        # Run inclination calculation
        inclination = self.calculate_angle(the_lines=pillar_edges)

        # TO DO: Check if metadata is available

        # Run cracks detection
        concrete_polygon = self.concrete_extractor.retrieve_polygon(the_lines=pillar_edges,
                                                                    image=pillar_subimage)

        cv2.imshow("cropped", concrete_polygon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        sys.exit()

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


    # def find_defects_pillars(self, pillars_detected, image, metadata):
    #     """
    #     Method performing defect detection (cracks, tilts) on concrete poles.
    #     :param pillars_detected: dictionary containing pillars detected
    #     :param image: image on which detection's happening
    #     :param metadata: this image's metadata (camera angles when this image was taken)
    #     :return:
    #     """
    #
    #     # ! Add attribute to an object reflecting its status and defect?! GOOD
    #
    #     tilt_detector = TiltCheckerOne()
    #     #tilt_detector = TiltCheckerTwo()
    #
    #     for pole_image_section, pillar in pillars_detected.items():
    #         # Pillar is a list. There must be only one object!
    #         pillar = pillar[0]
    #         # Create new subimage (use both coordinates of the pillar relative to the pole on which
    #         # it was detected and coordinates of the pole relative to the whole image. We do the same
    #         # when draw BBs
    #         pillar_subimage = np.array(image[pole_image_section.top + pillar.top:
    #                                          pole_image_section.top + pillar.bottom,
    #                                          pole_image_section.left + pillar.left:
    #                                          pole_image_section.left + pillar.right])
    #         # Find pole's edge (line, its coordinates) and the angle between this line and the bottom
    #         # edge of the image (no drone angle error considered at this point)
    #         the_line, tilt_angle = tilt_detector.check_pillar(pillar_subimage)
    #         if not the_line is None:
    #             line_relative = [the_line[0] + pole_image_section.left + pillar.left,
    #                              the_line[1] + pole_image_section.top + pillar.top,
    #                              the_line[2] + pole_image_section.left + pillar.left,
    #                              the_line[3] + pole_image_section.top + pillar.top]
    #
    #             return line_relative, tilt_angle
    #
    #         # HERE WE NEED TO TAKE INTO ACCOUNT THE ANGLES (ERRORS) USING
    #         # METADATA PROVIDED
    #
    #         # ! CRACKS DETECTION
    #
    # def find_defects_dumpers(self, components_detected):
    #     pass
    #
    # def find_defects_insulators(self, insulators_detected):
    #     pass
