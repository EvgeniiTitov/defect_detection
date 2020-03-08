import math
import numpy as np


class LineModifier:
    """
    Allows to merge short lines into longer ones provided they are
    similarly oriented.
    Allows to extend the lines representing concrete pole's edges in order to
    extract the area confined by these lines later
    """
    def __init__(self):

        self._angle_thresh = 70
        self._min_distance_to_merge = 30
        self._min_angle_to_merge = 30

    def extend_lines(
            self,
            lines_to_extend: list,
            image: np.ndarray
    ) -> list:
        """

        :param lines_to_extend:
        :return:
        """
        lines_extended = list()

        if len(lines_to_extend) == 2:

            for line in lines_to_extend:
                x1, y1 = line[0][0], line[0][1]
                x2, y2 = line[1][0], line[1][1]

                curr_lenght = math.sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2)

                # y = 0
                x_top = int(round(x1 + (x1 - x2) / curr_lenght * y1))

                # y = image.shape[0]
                x_bottom = int(
                    round(x2 + (x2 - x1) / curr_lenght * (image.shape[0] - y2))
                )
                # Dots are intentionally appended *flat* to the list, not typical
                # syntax (x1,y1), (x2,y2) etc
                # lines_extended.append([(x_top, 0), (x_bottom, image.shape[0])])
                lines_extended.append([x_top, 0])
                lines_extended.append([x_bottom, image.shape[0]])

        else:
            # If the algorithm failed to find two lines and returned only one,
            # we need to draw approximate second line to extract the area in between
            # First extend the line detected by the algorithm
            x1, y1 = lines_to_extend[0][0][0], lines_to_extend[0][0][1]
            x2, y2 = lines_to_extend[0][1][0], lines_to_extend[0][1][1]

            curr_lenght = math.sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2)

            # y = 0
            x_top = int(round(x1 + (x1 - x2) / curr_lenght * y1))

            # y = image.shape[0]
            x_bottom = int(
                round(x2 + (x2 - x1) / curr_lenght * (image.shape[0] - y2))
            )

            lines_extended.append([x_top, 0])
            lines_extended.append([x_bottom, image.shape[0]])

            # Draw second approximate line parallel to the first one.
            x_new_top = image.shape[1] - x_bottom
            x_new_bottom = image.shape[1] - x_top

            lines_extended.append([x_new_top, 0])
            lines_extended.append([x_new_bottom, image.shape[0]])

        return lines_extended

    def merge_lines(self, lines_to_merge: list) -> list:
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
            x2 = line[0][2]
            y2 = line[0][3]

            angle = abs(round(np.rad2deg(np.arctan2((y2 - y1), (x2 - x1))), 2))

            if angle < self._angle_thresh:
                continue

            vertical_lines.append([(x1, y1), (x2, y2)])

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

    def get_line_magnitude(self, x1, y1, x2, y2):

        line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))

        return line_magnitude

    def get_distance_point_line(self, px, py, x1, y1, x2, y2):

        # http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.get_line_magnitude(x1, y1, x2, y2)

        if LineMag < 0.00000001:
            distance_point_line = 9999

            return distance_point_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = self.get_line_magnitude(px, py, x1, y1)
            iy = self.get_line_magnitude(px, py, x2, y2)

            if ix > iy:
                distance_point_line = iy
            else:
                distance_point_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_line = self.get_line_magnitude(px, py, ix, iy)

        return distance_point_line

    def get_distance(self, line1, line2):

        dist1 = self.get_distance_point_line(line1[0][0], line1[0][1],
                                             line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.get_distance_point_line(line1[1][0], line1[1][1],
                                             line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.get_distance_point_line(line2[0][0], line2[0][1],
                                             line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.get_distance_point_line(line2[1][0], line2[1][1],
                                             line1[0][0], line1[0][1], line1[1][0], line1[1][1])

        return min(dist1, dist2, dist3, dist4)
