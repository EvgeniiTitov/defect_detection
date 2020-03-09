import numpy as np
import cv2
import sys


class ConcreteExtractor:
    """
    Allows to find pole edges and extract the area confined by these edges
    """
    def __init__(
            self,
            line_modifier,
    ):

        self.line_modifier = line_modifier

    def retrieve_polygon_v2(
            self,
            image: np.ndarray,
            the_edges: list,
            width: int=224,
            height: int=1200
    ):
        """

        :param image:
        :param the_edges:
        :param width:
        :param height:
        :return:
        """
        extended_lines = self.line_modifier().extend_lines(lines_to_extend=the_edges,
                                                           image=image)

        # Change coordinates order as per warp perspective requirements
        extended_lines.append(extended_lines.pop(extended_lines.index(extended_lines[1])))

        points = np.array(extended_lines, dtype="float32")
        dim = (width, height)

        return cv2.resize(self.wrap_perspective(image, points),
                          dim,
                          interpolation=cv2.INTER_AREA)

    def wrap_perspective(
            self,
            image,
            points):
        """
        Rebuild image based on the points provided - make it look bird-like view
        :param image:
        :param points:
        :return:
        """
        top_left, top_right, bot_right, bot_left = points

        # Find max distance
        width_1 = np.sqrt(((bot_right[0] - bot_left[0]) ** 2) + ((bot_right[1] - bot_left[1]) ** 2))
        width_2 = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        max_width = max(int(width_1), int(width_2))

        # Compute new height
        height_1 = np.sqrt(((top_right[0] - bot_right[0]) ** 2) + ((top_right[1] - bot_right[1]) ** 2))
        height_2 = np.sqrt(((top_left[0] - bot_left[0]) ** 2) + ((top_left[1] - bot_left[1]) ** 2))
        max_height = max(int(height_1), int(height_2))

        # Build set of destination top-down view like points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        transform = cv2.getPerspectiveTransform(src=points, dst=dst)
        warped = cv2.warpPerspective(src=image, M=transform, dsize=(max_width, max_height))

        return warped

    def retrieve_polygon(
            self,
            the_lines: list,
            image: np.ndarray
    ) -> np.ndarray:
        """
        :param the_lines:
        :return:
        """
        # Since lines are usually of varying length and almost always are
        # shorter than image's height, extend them first to successfully extract
        # the area confined by them

        extended_lines = self.line_modifier().extend_lines(lines_to_extend=the_lines,
                                                           image=image)

        # Once line's been extended, use them to extract the image section
        # restricted, defined by them
        support_point = extended_lines[2]
        extended_lines.append(support_point)

        points = np.array(extended_lines)

        mask = np.zeros((image.shape[0], image.shape[1]))

        # Fills  in the shape defined by the points to be white in the mask. The
        # rest is black
        cv2.fillConvexPoly(img=mask,
                           points=points,
                           color=1)

        # We then convert the mask into Boolean where white pixels refrecling
        # the image section we want to extract as True, the rest is False
        mask = mask.astype(np.bool)

        # Create a white empty image
        output = np.zeros_like(image)

        # Use the Boolean mask to index into the image to extract out the pixels
        # we need. All pixels that happened to be mapped as True are taken
        output[mask] = image[mask]

        output_copy = output.copy()

        # Get indices of all pixels that are black
        black_pixels_indices = np.all(output == [0, 0, 0], axis=-1)
        # Invert the matrix to get indices of not black pixels
        non_black_pixels_indices = ~black_pixels_indices

        # All black pixels become white, all not black pixels get their original values
        output_copy[black_pixels_indices] = [255, 255, 255]
        output_copy[non_black_pixels_indices] = output[non_black_pixels_indices]

        return output_copy

    def find_pole_edges(self, image: np.ndarray) -> list:
        """
        Find pole's edges by means of generating lines (Canny, HoughTrans) ands .

        :return: edges found (list of lists)
        """
        # Find all lines on the image
        raw_lines = self.generate_lines(image)

        # Rewrite lines in a proper form (x1,y1), (x2,y2) if any found. List of lists
        if raw_lines is None:
            return []

        # Process results: merge raw lines where possible to decrease the total
        # number of lines we are working with
        merged_lines = self.line_modifier().merge_lines(lines_to_merge=raw_lines)

        # Pick lines based on which the angle will be calculated. Ideally we are looking for 2 lines
        # which represent both pole's edges. If there is 1, warn user and calculate the angle based
        # on it. Pick two opposite and parrallel lines within the merged ones. We assume this is pole
        if len(merged_lines) > 1:
            the_lines = self.retrieve_pole_lines(merged_lines, image)

        elif len(merged_lines) == 1:
            print("WARNING: Only one edge detected!")
            the_lines = merged_lines

        else:
            print("WARNING: No edges detected")
            return []

        assert the_lines and 1 <= len(the_lines) <= 2, "ERROR: Wrong number of lines found"

        return the_lines

    def retrieve_pole_lines(
            self,
            merged_lines: list,
            image: np.ndarray
    ) -> list:
        """
        Performs all sorts of filtering to pick only 2 lines - the ones that most likely
        going to pole's edges.
        :param merged_lines: Lines detected (list of lists)
        :param image: image getting processed
        :return: lines (list of list(s))
        """

        # Sort all lines based on their position relatively to imaginary dividing line
        # in the middle of the image. We allow 10% margin along the dividing line to account
        # for lines which might have a point slightly shifted to the *wrong* side along X axis
        import math
        lines_to_the_left = list()
        lines_to_the_right = list()
        left_section_and_margin = int(image.shape[1] * 0.6)
        right_section_and_margin = int(image.shape[1] * 0.4)

        if len(merged_lines) > 10:
            print("WARNING: MORE THAN 10 LINES TO SORT. O(N2) WONT PROMISE YOU THAT")

        while merged_lines:

            line = merged_lines.pop()
            line_angle = round(90 - np.rad2deg(np.arctan2(abs(line[1][1] - line[0][1]),
                                                          abs(line[1][0] - line[0][0]))), 2)
            line_lenght = math.sqrt((line[1][1] - line[0][1])**2 + (line[1][0] - line[0][0])**2)

            if line[0][0] <= left_section_and_margin and line[1][0] <= left_section_and_margin:
                lines_to_the_left.append((line, line_angle, line_lenght))
                # to make sure the same line doesn't get added to both subgroups if it lies in the margin
                continue

            if line[0][0] >= right_section_and_margin and line[1][0] >= right_section_and_margin:
                lines_to_the_right.append((line, line_angle, line_lenght))

        # Pick 2 best lines (2 most parallel)
        # O(n2). Slow, but we do not deal with large number of lines anyway
        optimal_lines = 180, None, None  # angle difference, line 1, line 2

        # Possible that the whole pole lies in the left part of the image
        if lines_to_the_left and not lines_to_the_right:
            # Select only among the lines to the left
            if len(lines_to_the_left) == 1:
                # Return only coordinates without angle and lenght
                return [lines_to_the_left[0][0]]

            elif len(lines_to_the_left) == 2:
                # Check if both lines to the left are relatively parallel -> pole
                if abs(lines_to_the_left[0][1] - lines_to_the_left[1][1]) <= 2:
                    return [lines_to_the_left[0][0], lines_to_the_left[1][0]]
                # Else return the longest one - likely to be pole's edge + some noise
                else:
                    return [lines_to_the_left[0][0]] if lines_to_the_left[0][2] > lines_to_the_left[1][2] else\
                           [lines_to_the_left[1][0]]

            # Have more than 2 lines to the left. Need to find the 2
            else:
                for i in range(len(lines_to_the_left) - 1):
                    for j in range(i + 1, len(lines_to_the_left)):

                        delta = abs(lines_to_the_left[i][1] - lines_to_the_left[j][1])

                        if not delta < optimal_lines[0]:
                            continue
                        else:
                            optimal_lines = delta, lines_to_the_left[i][0], lines_to_the_left[j][0]

        # Possible that the whole pole lies in the right part of the image
        elif lines_to_the_right and not lines_to_the_left:
            # Select only among the lines to the right
            if len(lines_to_the_right) == 1:
                return [lines_to_the_right[0][0]]

            elif len(lines_to_the_right) == 2:
                # Check if both lines to the right are relatively parallel -> pole
                if abs(lines_to_the_right[0][1] - lines_to_the_right[1][1]) <= 2:
                    return [lines_to_the_right[0][0], lines_to_the_right[1][0]]
                else:
                    return [lines_to_the_right[0][0]] if lines_to_the_right[0][2] > lines_to_the_right[1][2] else\
                           [lines_to_the_right[1][0]]

            else:
                for i in range(len(lines_to_the_right) - 1):
                    for j in range(i + 1, len(lines_to_the_right)):

                        delta = abs(lines_to_the_right[i][1] - lines_to_the_right[j][1])

                        if not delta < optimal_lines[0]:
                            continue
                        else:
                            optimal_lines = delta, lines_to_the_right[i][0], lines_to_the_right[j][0]

        # Ideal case - lines are to the left and to the rest. Find the best 2 (most parallel ones)
        else:
            for left_line, left_angle, left_length in lines_to_the_left:
                for right_line, right_angle, right_length in lines_to_the_right:

                    delta = abs(left_angle - right_angle)

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
        # rect = (int(image.shape[1] * 0.1),
        #         0,
        #         image.shape[1] - int(image.shape[1] * 0.2),
        #         image.shape[0])

        rect = (1,
                0,
                image.shape[1],
                image.shape[0])

        cv2.grabCut(image,
                    mask,
                    rect,
                    bgd_model,
                    fgd_model,
                    5,
                    cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        img = image * mask2[:, :, np.newaxis]

        ret, thresh = cv2.threshold(img,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY)

        return thresh
