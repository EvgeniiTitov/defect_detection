import math
import cv2
import os
import sys
import numpy as np


class TiltDetector:
    """
    By default we do not need a results handler because once the module gets integrated
    into a larger system it won't be tasked with any other tasks but calculating tilt angle
    and returning it. This is why result processor by default have been set to None.
    """
    def __init__(
            self,
            results_handling_way,
            line_merger,
            results_processor=None
    ):

        # Results processing approach
        self.results_processing_flag, self.save_path = results_handling_way
        self.merger = line_merger

        if self.results_processing_flag:
            self.processer = results_processor

        self._line_thickness = 2
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._font_colour = (255, 255, 255)
        self._line_type = 3

    def find_pole_edges(self, image):
        """Runs an image provided along the pole inclination angle
        calculating pipeline.

        :param path_to_image: path to image
        :return: angle calculated if any pole edges found. Else returns None
        """
        # Find all lines on the image
        raw_lines = self.generate_lines(image)

        # Rewrite lines in a proper form (x1,y1), (x2,y2) if any found. List of lists
        if raw_lines is not None:
            lines_to_merge = list()
            for line in self.get_lines(raw_lines):
                lines_to_merge.append(
                            [(line[0], line[1]), (line[2], line[3])]
                                      )
        else:
            return None, None

        # Process results: merge raw lines where possible to decrease the total
        # number of lines we are working with
        merged_lines = self.merger.merge_lines(lines_to_merge)

        # VISUALIZE MERGED LINES
        #self.processer.save_image(merged_lines, image, image_name, len(merged_lines))

        # Pick lines based on which the angle will be calculated. Ideally we are looking for 2 lines
        # which represent both pole's edges. If there is 1, warn user and calculate the angle based
        # on it. Pick two opposite and parrallel lines within the merged ones. We assume this is pole
        if len(merged_lines) > 1:
            the_lines = self.retrieve_pole_lines(merged_lines, image)

        elif len(merged_lines) == 1:
            print("WARNING: Only one line found, angle will be calculated based on it")
            the_lines = merged_lines

        else:
            print("No lines found, angle cannot be calculated")
            return

        assert the_lines and len(the_lines) == 1 or len(the_lines) == 2, "ERROR: Wrong number of lines found"


        # if self.results_processing_flag:
        #     self.processer.save_image(the_lines,
        #                               image,
        #                               image_name,
        #                               angle)

        return the_lines

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

    def retrieve_pole_lines(self, merged_lines, image):
        """
        Among all lines found we need to pick only 2 - the ones that most likely
        going to pole's edges
        :param merged_lines: Lines detected (list of lists)
        :param image: image getting processed
        :return: 2 lines (list of lists)
        """

        # Sort all lines based on their position relatively to imaginary dividing line
        # in the middle of the image. We allow 5% margin along the dividing line to account
        # for lines which might have a point slightly shifted to the *wrong* side along X axis
        lines_to_the_left = list()
        lines_to_the_right = list()
        left_section_and_margin = int(image.shape[1] * 0.6)
        right_section_and_margin = int(image.shape[1] * 0.4)

        while merged_lines:
            line = merged_lines.pop()

            if line[0][0] <= left_section_and_margin and line[1][0] <= left_section_and_margin:
                lines_to_the_left.append(line)
                continue  # to make sure the same line doesn't get added to both

            if line[0][0] >= right_section_and_margin and line[1][0] >= right_section_and_margin:
                lines_to_the_right.append(line)

        # Pick 2 best lines (2 most parallel)
        # O(n2). Slow, but we do not deal with large number of lines anyway
        optimal_lines = 180, None, None # angle difference, line 1, line 2

        for left_line in lines_to_the_left:

            x1 = left_line[0][0]
            y1 = left_line[0][1]
            x2 = left_line[1][0]
            y2 = left_line[1][1]
            left_line_angle = round(90 - np.rad2deg(np.arctan2(abs(y2 - y1), abs(x2 - x1))), 2)

            for right_line in lines_to_the_right:

                x1_1 = right_line[0][0]
                y1_1 = right_line[0][1]
                x2_2 = right_line[1][0]
                y2_2 = right_line[1][1]
                right_line_angle = round(90 - np.rad2deg(np.arctan2(abs(y2_2 - y1_1), abs(x2_2 - x1_1))), 2)

                delta = abs(left_line_angle - right_line_angle)

                if not delta < optimal_lines[0]:
                    continue

                optimal_lines = delta, left_line, right_line

        return [optimal_lines[1], optimal_lines[2]]

    def generate_lines(self, image):
        """Generates lines based on which the inclination angle will be
        later calculated
        :param image: image
        :return: image with generated lines
        """
        # Apply mask to remove background
        image_masked = self.apply_mask(image)

        # Generate edges
        edges = cv2.Canny(image_masked,
                          threshold1=50,
                          threshold2=200,
                          apertureSize=3)
        # Based on the edges found, find lines
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=100,
                                minLineLength=100,
                                maxLineGap=100)

        return lines

    def apply_mask(self, image):
        """
        Applies rectangular mask to an image in order to remove background
        and mainly focus on the pole
        :param image: original image
        :return: image with the mask applied
        """
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # start_x, start_y, width, height
        rect = (int(image.shape[1] * 0.1),
                0,
                image.shape[1] - int(image.shape[1] * 0.2),
                image.shape[0])

        cv2.grabCut(image,
                    mask,
                    rect,
                    bgd_model,
                    fgd_model,
                    10,
                    cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img = image * mask2[:, :, np.newaxis]

        ret, thresh = cv2.threshold(img,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY)

        return thresh

    def get_lines(self, lines_in):
        if cv2.__version__ < "3.0":
            return lines_in[0]

        return [line[0] for line in lines_in]


class LineMerger:
    """
    Allows to merge shorter lines into longer ones provided they are
    similarly oriented.
    In addition, discards all non-vertical lines (not with 40 degrees
    vertical cone
    """
    def __init__(self):
        self._angle_thresh = 70
        self._min_distance_to_merge = 30
        self._min_angle_to_merge = 30

    def merge_lines(self, lines_to_merge):
        """
        Merges lines provided they are similarly oriented
        :param lines_to_merge: List of lists. Lines to merge
        :return: merged lines
        """
        vertical_lines = self.discard_not_vertical_lines(lines_to_merge)

        # Sort and get line orientation
        lines_x, lines_y = list(), list()

        for line in vertical_lines:
            orientation = math.atan2(
                    (line[0][1] - line[1][1]), (line[0][0] - line[1][0])
                                     )

            if (abs(math.degrees(orientation)) > 45) and\
                                abs(math.degrees(orientation)) < (90 + 45):
                lines_y.append(line)
            else:
                lines_x.append(line)

        lines_x.sort(key=lambda line: line[0][0])
        lines_y.sort(key=lambda line: line[0][1])

        merged_lines_x = self.merge_lines_pipeline_2(lines_x)
        merged_lines_y = self.merge_lines_pipeline_2(lines_y)

        merged_lines_all = list()
        merged_lines_all.extend(merged_lines_x)
        merged_lines_all.extend(merged_lines_y)

        return merged_lines_all

    def discard_not_vertical_lines(self, lines):
        """
        Discards all lines that are not within N degrees cone
        :param lines:
        :return: vertical lines [(point1, point2, angle of the line), ]
        """
        # Discard horizontal lines (no point merging lines that are not what we need)
        vertical_lines = list()

        for line in lines:

            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]

            angle = abs(round(np.rad2deg(np.arctan2((y2 - y1), (x2 - x1))), 2))

            if angle < self._angle_thresh:
                continue

            vertical_lines.append(
                        [(line[0][0], line[0][1]), (line[1][0], line[1][1])]
                                  )

        return vertical_lines

    def merge_lines_pipeline_2(self, lines):
        """

        :param lines:
        :return:
        """
        super_lines_final = []
        super_lines = []

        # check if a line has angle and enough distance to be considered similar
        for line in lines:
            create_new_group = True
            group_updated = False

            for group in super_lines:
                for line2 in group:

                    if self.get_distance(line2, line) < self._min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < self._min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))
                            group.append(line)

                            create_new_group = False
                            group_updated = True
                            break

                if group_updated:
                    break

            if create_new_group:
                new_group = list()
                new_group.append(line)

                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < self._min_distance_to_merge:
                        # check the angle between lines
                        orientation_i = math.atan2((line[0][1] - line[1][1]), (line[0][0] - line[1][0]))
                        orientation_j = math.atan2((line2[0][1] - line2[1][1]), (line2[0][0] - line2[1][0]))

                        if int(abs(abs(math.degrees(orientation_i)) - abs(
                                math.degrees(orientation_j)))) < self._min_angle_to_merge:
                            # print("angles", orientation_i, orientation_j)
                            # print(int(abs(orientation_i - orientation_j)))

                            new_group.append(line2)

                            # remove line from lines list
                            # lines[idx] = False
                # append new group
                super_lines.append(new_group)

        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments1(group))

        return super_lines_final

    def merge_lines_segments1(self, lines, use_log=False):

        if len(lines) == 1:
            return lines[0]

        line_i = lines[0]

        # orientation
        orientation_i = math.atan2((line_i[0][1] - line_i[1][1]), (line_i[0][0] - line_i[1][0]))

        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])

        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90 + 45):

            # sort by y
            points = sorted(points, key=lambda point: point[1])

            if use_log:
                print("use y")
        else:

            # sort by x
            points = sorted(points, key=lambda point: point[0])

            if use_log:
                print("use x")

        return [points[0], points[len(points) - 1]]

    def lines_close(self, line1, line2):

        dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
        dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
        dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
        dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

        if min(dist1, dist2, dist3, dist4) < 100:
            return True
        else:
            return False

    def lineMagnitude(self, x1, y1, x2, y2):

        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

        return lineMagnitude

    def DistancePointLine(self, px, py, x1, y1, x2, y2):
        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.lineMagnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = self.lineMagnitude(px, py, x1, y1)
            iy = self.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = self.lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, line1, line2):

        dist1 = self.DistancePointLine(line1[0][0], line1[0][1],
                                       line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.DistancePointLine(line1[1][0], line1[1][1],
                                       line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.DistancePointLine(line2[0][0], line2[0][1],
                                       line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.DistancePointLine(line2[1][0], line2[1][1],
                                       line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)
