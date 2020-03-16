from defect_detectors import ConcreteExtractor, LineModifier
import threading
import numpy as np


"""
Inside defect detector you can have N number of threads each tasked with
finding defects on an object of particular class. Connected by Qs, return
results to one place (join threads), which get put in Q_out and sent for postprocessing

VS multiprocessing. Might be faster but data transfer overhead
"""



class DefectDetector(threading.Thread):

    def __init__(
            self,
            queue_from_object_detector,
            queue_to_results_processor,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_object_detector
        self.Q_out = queue_to_results_processor

        self.concrete_extractor = ConcreteExtractor(line_modifier=LineModifier)

        # self.cracks_tester = cracks_detector
        # self.dumpers_tester = dumpers_defect_detector
        # self.insulators_tester = insulators_defect_detector

        print("Defect detectors initialized")

    def run(self) -> None:

        while not self.done:

            detected_defects = {
                "pillar": [],
                "dumper": [],
                "insulator": []
            }

            # Get a dictionary of detected objects
            item = self.Q_in.get(block=True)

            if item == "END":
                self.Q_out.put("END")
                break

            image, detected_items = item

            for subimage, elements in detected_items.items():
                for index, element in enumerate(elements):

                    if element.object_name.lower() == "pillar":
                        # Doesn't return anything, changes object's state
                        self.find_defects_pillar(pillar=element,
                                                 subimage_section=subimage)

                        pillar_status = {
                            "inclination": element.inclination,
                            "cracked": element.cracked
                        }

                        detected_defects["pillar"].append(pillar_status)

                    elif element.object_name.lower() == "dump":
                        continue

                    elif element.object_name.lower() == "insl":
                        continue

                    else:
                        continue

            self.Q_out.put((image, detected_defects, detected_items))

        return

    def find_defects_pillar(
            self,
            pillar,
            subimage_section
    ):
        # Reconstruct pillar bounding box image
        pillar_bb = np.array(subimage_section.frame[pillar.top:pillar.bottom,
                                                    pillar.left:pillar.right])

        pillar_edges = self.concrete_extractor.find_pole_edges(image=pillar_bb)

        if not pillar_edges:
            return

        inclination = self.calculate_angle(the_lines=pillar_edges)

        if inclination and 0 <= inclination <= 90:
            pillar.inclination = inclination
        else:
            pillar.inclination = "NULL"

        # TODO: Crack detection 1.Extract polygon and send for analysis
        # TODO: change object's state to .cracked True or False

    def find_defects_dumpers(
            self,
            dumper,
            subimage_section
    ):
        pass

    def find_defects_insulators(
            self,
            insulator,
            subimage_section
    ):
        pass

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
