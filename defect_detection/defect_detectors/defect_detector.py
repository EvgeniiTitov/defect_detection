from .pole_tilt_checker import TiltCheckerOne, TiltCheckerTwo
import numpy as np


class DefectDetector:
    def __init__(self, defects):
        # We want to initialize only those detectors that we need!
        if defects:
            for component, detecting_flag in defects.items():
                if detecting_flag and component == "pole_defects":
                    self.tilt_checker = TiltCheckerOne()
                elif detecting_flag and component == "dumper_defects":
                    print("Initialized dumper defect detectors")
                elif detecting_flag and component == "insulator_defects":
                    print("Initialized insulator defect detectors")

    def find_defects_pillars(self, pillars_detected, image, metadata):
        """
        Method performing defect detection (cracks, tilts) on concrete poles.
        :param pillars_detected: dictionary containing pillars detected
        :param image: image on which detection's happening
        :param metadata: this image's metadata (camera angles when this image was taken)
        :return:
        """

        # ! Add attribute to an object reflecting its status and defect?! GOOD

        tilt_detector = TiltCheckerOne()
        #tilt_detector = TiltCheckerTwo()

        for pole_image_section, pillar in pillars_detected.items():
            # Pillar is a list. There must be only one object!
            pillar = pillar[0]
            # Create new subimage (use both coordinates of the pillar relative to the pole on which
            # it was detected and coordinates of the pole relative to the whole image. We do the same
            # when draw BBs
            pillar_subimage = np.array(image[pole_image_section.top + pillar.top:
                                             pole_image_section.top + pillar.bottom,
                                             pole_image_section.left + pillar.left:
                                             pole_image_section.left + pillar.right])
            # Find pole's edge (line, its coordinates) and the angle between this line and the bottom
            # edge of the image (no drone angle error considered at this point)
            the_line, tilt_angle = tilt_detector.check_pillar(pillar_subimage)
            if not the_line is None:
                line_relative = [the_line[0] + pole_image_section.left + pillar.left,
                                 the_line[1] + pole_image_section.top + pillar.top,
                                 the_line[2] + pole_image_section.left + pillar.left,
                                 the_line[3] + pole_image_section.top + pillar.top]

                return line_relative, tilt_angle

            # HERE WE NEED TO TAKE INTO ACCOUNT THE ANGLES (ERRORS) USING
            # METADATA PROVIDED

            # ! CRACKS DETECTION

    def find_defects_dumpers(self, components_detected):
        pass

    def find_defects_insulators(self, insulators_detected):
        pass
