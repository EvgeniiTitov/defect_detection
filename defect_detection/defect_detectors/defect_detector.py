from .pole_tilt_checker import TiltChecker
import numpy as np


class DefectDetector:
    def __init__(self, defects):
        # We want to initialize only those detectors that we need!
        if defects:
            for component, detecting_flag in defects.items():
                if detecting_flag and component == "pole_defects":
                    self.tilt_checker = TiltChecker()
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

        # ! SOMETHING TO STORE DEFECTS DETECTED

        for pole_image_section, pillar in pillars_detected.items():
            # Pillar is a list. There must be only one object!
            pillar = pillar[0]
            # Create new subimage (use both coordinates of the pillar relative to the pole on which
            # it was detected and coordinates of the pole relative to the whole image. We do the same
            # when draw BBs
            pillar_subimage = np.array(image[pole_image_section.top + pillar.BB_top:
                                             pole_image_section.top + pillar.BB_bottom,
                                             pole_image_section.left + pillar.BB_left:
                                             pole_image_section.left + pillar.BB_right])

            if metadata:
                # Then we can perform tilt detection
                pitch_angle, roll_angle = metadata
                tilt_detector = TiltChecker()
                result = tilt_detector.check_pillar(pillar_subimage,
                                                    pitch=pitch_angle,
                                                    roll=roll_angle)

            # ! CRACKS DETECTION

    def find_defects_dumpers(self, components_detected):
        pass

    def find_defects_insulators(self, insulators_detected):
        pass
