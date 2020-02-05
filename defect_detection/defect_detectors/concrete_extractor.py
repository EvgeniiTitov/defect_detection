import numpy as np
import cv2


class ConcreteExtractor:
    """
    Allows to find pole edges and extract the area confined by these edges
    """
    def __init__(
            self,
            line_modifier,
    ):

        self.line_modifier = line_modifier

    def retrieve_polygon(self, the_lines, image):
        """

        :param the_lines:
        :return:
        """
        # Since lines are usually of varying length and almost always are
        # shorter than image's height, extend them first to successfully extract
        # the area confined by them
        extended_lines = list()

        # To check when just one line was detected
        # the_lines = [the_lines[1]]

        extended_lines += self.line_modifier().extend_lines(lines_to_extend=the_lines,
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

    def find_pole_edges(self, image):
        """Runs an image provided along the pole inclination angle
        calculating pipeline.

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
        merged_lines = self.line_modifier().merge_lines(lines_to_merge=lines_to_merge)

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

        return the_lines

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
        optimal_lines = 180, None, None  # angle difference, line 1, line 2

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
