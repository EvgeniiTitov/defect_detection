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
            cracks_detector=None,
            dumpers_defect_defector=None,
            insulators_defect_detector=None,
            camera_orientation=None
    ):

        self.camera_orientation = camera_orientation

        # Actual defect detecting modules
        self.cracks_tester = cracks_detector
        self.dumper_tester = dumpers_defect_defector
        self.insulator_tester = insulators_defect_detector

        self.auxiliary_modules_initialized = False

        # Cache the lines once they have been found
        self.the_lines = None

    def search_defects(
            self,
            detected_objects,
    ):
        """

        :param detected_objects:
        :return:
        """
        # Subimage is either a subimage of a pole if any have been detected or the whole original
        # image on case no poles were found. Elements are objects detected within this subimage
        for subimage, elements in detected_objects.items():

            for element in elements:
                if element.object_name.lower() == "pillar":

                    # Do not return anything. Change object's state - declare it defected or not
                    self.pillars_defects_detector(pillar=element,
                                                  detection_section=subimage)

                elif element.object_name.lower() == "dump":
                    # Search for defects on vibration dumpers
                    continue

                elif element.object_name.lower() == "insul":
                    # Search for defects on insulators
                    continue

    def pillars_defects_detector(
            self,
            pillar,
            detection_section
    ):
        """
        Pipeline for detecting pillars inclination and cracks
        :param pillar:
        :param detection_section:
        :return:
        """
        # Check if inclination and cracks detectors have been already initialized
        if not self.auxiliary_modules_initialized:
            line_modifier = LineModifier(image=detection_section)
            concrete_extractor = ConcreteExtractor(image=detection_section,
                                                   line_modifier=line_modifier)

            self.auxiliary_modules_initialized = True

        # Find the edges first as a separate task





        component_subimage = np.array(detection_section.frame[pillar.top:pillar.bottom,
                                                              pillar.left:pillar.right])

        #

        cv2.imshow("cropped", )
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
