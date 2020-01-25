import numpy as np
from collections import defaultdict
from .inclination_detector import TiltDetector, LineMerger
from .concrete_polygon_extractor import LineExtender, PolygonRetriever

from .concrete_extractor import ConcreteExtractor
from .line_modifier import LineModifier
import sys
import cv2

#How to make it scalabable? It needs to be given *tools to use*. Like give it a detector for each defect,
#Keep interfaces the same - give image, receive some data structure that says if there are any defects found


class DefectDetector:
    """
    TO DO: Consider multiprocessing
    """
    def __init__(self, camera_orientation=None):

        self.camera_orientation = camera_orientation

    def initialize_detector(self):

        pass

    def search_defects(
            self,
            detected_objects,
    ):
        """

        :param detected_objects:
        :return:
        """
        defects = defaultdict(list)

        for detection_image_section, elements in detected_objects.items():

            for element in elements:

                if element.object_name.lower() == "pillar":

                    # Go ahead and search for cracks and pole inclination defects
                    defects = self.pillars_defects_detector(pillar=element,
                                                            detection_section=detection_image_section,
                                                            metadata=metadata)

                elif element.object_name.lower() == "dump":
                    continue

                elif element.object_name.lower() == "insul":
                    continue

    def pillars_defects_detector(
            self,
            pillar,
            detection_section,
            metadata
    ):

        component_subimage = np.array(detection_section.frame[pillar.top:pillar.bottom,
                                                              pillar.left:pillar.right])

        # Angle calculations
        the_edges = self.inclination_detector.find_pole_edges(image=component_subimage)
        angle = self.inclination_detector.calculate_angle(the_lines=the_edges)

        if metadata:
            # TAKE INTO ACCOUNT METADATA (CAMERA MIGHT HAVE BEEN TILTED)
            pass

        # Cracks detection
        concrete_polygon = self.concrete_polygon_extractor.retrieve_polygon(image=component_subimage,
                                                                            the_lines=the_edges)
        # ONCE WE'VE GOT POLYGON, SEND IT FOR CRACK DETECTION

        print("angle: ", angle)

        cv2.imshow("cropped", concrete_polygon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        sys.exit()






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
