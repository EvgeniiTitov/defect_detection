import cv2, sys
import numpy as np

class TiltCheckerOne:
    """
    Tilt checking class. Find 5 longest lines, then among those lines it searches
    for the most vertical line. This leads to issues when wrong edge gets detected
    like a wire which sometimes happens to appear more vertical than a pole
    """
    def __init__(self,
                 min_line_lenght=100,
                 max_line_gap=200,
                 tilt_threshold=3,
                 resize_coef=1):

        self.min_line_lenght = min_line_lenght
        self.max_line_gap = max_line_gap
        self.tilt_thresh = tilt_threshold
        self.resize_coef = resize_coef

    def check_pillar(self, image):
        """
        :param image: numpy array. Image section containing a pole cropped
        :return: angle of inclination, pole status (green, yellow, red)
        """
        # Resize image
        if self.resize_coef == 1:
            resized = image
        else:
            resized = self.resize_image(image)
        # Modify the image (filters), find Canny edges
        modified_image = self.modify_image(resized)
        # Find all lines on the image
        lines = self.extract_lines(modified_image)
        # Check if any lines have been detected
        if lines is None:
            print("No lines detected")
            return
        # Find N (5) longest lines
        longest_lines = self.N_longest_lines(lines)
        # Among those 5 lines find the most vertical line - the line
        the_line = self.find_most_vertical_line(longest_lines)[0]
        # Calculate angle between the line and the bottom edge of an image
        angle_rel_to_horizon = self.calculate_angle(the_line)
        # Convert the calculations to be relatively to the vertical line
        tilt_angle = 90 - angle_rel_to_horizon

        return the_line, tilt_angle

    def calculate_angle(self, line):
        """
        Calculate angle between the line found and image's bot
        :param line:
        :return:
        """
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        angle = np.rad2deg(np.arctan2(abs(y2-y1),abs(x2-x1)))

        return angle

    def resize_image(self, image):
        """
        :param image:
        :return:
        """
        # Resize image
        width = int(image.shape[1] * self.resize_coef)
        height = int(image.shape[0] * self.resize_coef)
        dim = (width, height)
        resized_image = cv2.resize(image,
                                   dim,
                                   interpolation=cv2.INTER_AREA)

        return resized_image

    def modify_image(self, image):
        """
        Modified image. Find edges
        :param image:
        :return: image modified
        """
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Blur the image
        blurred_image = cv2.bilateralFilter(gray_image, 3, 100, 100)  # 3,75,75?
        # Apply Canny edge detector
        canny = cv2.Canny(blurred_image, 35, 400)

        return canny

    def extract_lines(self, image):
        """
        Extracts lines longer than a certain threshold in the image provided
        :param image:
        :return:
        """
        lines = cv2.HoughLinesP(image=image,
                                rho=1,  # Distance resolution of accumulator in pixels
                                theta=np.pi/180,  # Angle resolution of the accumulator (radians)
                                threshold=100,  # Accumulator threshold parameter. Only lines that got enough votes get returned
                                lines=np.array([]),  # Output vector of lines. Each line (x1, y1, x2, y2)
                                minLineLength=self.min_line_lenght,  # Min line lenght. Segments shorter are rejected
                                maxLineGap=self.max_line_gap)  # Max allowed gap between points on the same line to link them

        return lines

    def N_longest_lines(self, lines):
        """
        NOTE! TIME COMPLEXITY O(n)!
        Calculates and returns coordinates of the N longest line
        :param lines:
        :return:
        """
        longest_lines = list()
        for index, line_coordinates in enumerate(lines):
            # Extra index since its a list in the list
            x1 = line_coordinates[0][0]
            y1 = line_coordinates[0][1]
            x2 = line_coordinates[0][2]
            y2 = line_coordinates[0][3]
            line_lenght = ((x2-x1)**2 + (y2-y1)**2)**0.5
            longest_lines.append((index, line_lenght))

        # Sort the list by lenght (longest go first)
        longest_lines.sort(key=lambda e: e[-1], reverse=True)
        # Collect indices of 5 longest lines from the list above (its been sorted,
        # so they are not in order anymore)
        indices_of_longest = [i for i, lenght in longest_lines]

        # Return 5 longest lines from the original list using the indices of the longest ones
        if len(longest_lines) >= 5:
            return [lines[i] for i in indices_of_longest[:5]]
        else:
            return [lines[i] for i in indices_of_longest]

    def find_most_vertical_line(self, lines):
        """
        Time complexity O(n)!
        Calculates the most vertical line among all the lines found
        :param lines:
        :return:
        """
        vertical_line = (0,180)  # index, angle
        for index, line_coordinates in enumerate(lines):
            # Extra index since its a list in the list
            x1 = line_coordinates[0][0]
            y1 = line_coordinates[0][1]
            x2 = line_coordinates[0][2]
            y2 = line_coordinates[0][3]
            angle = np.rad2deg(np.arctan2((y2-y1),(x2-x1)))
            if 90 - abs(angle) < vertical_line[-1]:
                vertical_line = (index, 90 - abs(angle))

        return lines[vertical_line[0]]


class TiltCheckerTwo:
    """
    Tilt checker two. This implementation detects all the lines. Then, it searches for the
    lines that have the same orientation angle (attempt to properly detect a pole).
    """
    def __init__(self,
                 min_line_lenght=100,
                 max_line_gap=200,
                 tilt_threshold=3,
                 resize_coef=1):

        self.min_line_lenght = min_line_lenght
        self.max_line_gap = max_line_gap
        self.tilt_thresh = tilt_threshold
        self.resize_coef = resize_coef

    def check_pillar(self, image):
        """
        :param image: numpy array. Image section containing a pole cropped
        :return: angle of inclination, pole status (green, yellow, red)
        """
        # Resize image
        if self.resize_coef == 1:
            resized = image
        else:
            resized = self.resize_image(image)
        # Modify the image (filters), find Canny edges
        modified_image = self.modify_image(resized)
        # Find all lines on the image
        lines = self.extract_lines(modified_image)
        # Check if any lines have been detected
        if lines is None:
            print("No lines detected")
            return

        # 1. Find lines with the same angle
        # 2. If this angle is less than the threshold (we do not need horizontal lines,
        #    so maybe search for lines > 45 degrees), discard the lines. Check for another
        #    set of lines. Introduce some leeway so the lines do not have to be precisely of
        #    the same angle. 1 degree variation?
        # 3. If multiple parallel lines found, choose the longest ones because it is likely
        #    going to be a pole
        # 4. If no lines have been found, try another set of Canny and edge detecting parameters
        vertical_lines = self.vertical_lines(lines)

        if len(vertical_lines) == 0:
            print("No vertical lines detected")
            return
        print("N of vertical lines found:", len(vertical_lines))

        # Among those vertical lines discard short ones (get 6 longest or if less keep all)
        longest_lines = self.get_N_longest_lines(vertical_lines)
        # Among those longest, search for parallel lines (to find a pole)
        the_lines = self.parallel_lines(longest_lines)

        self.draw_lines(the_lines, resized)
        # ERROR BECAUSE NEEDS TO RETURN TWO VALUES

    def resize_image(self, image):
        """
        :param image:
        :return:
        """
        # Resize image
        width = int(image.shape[1] * self.resize_coef)
        height = int(image.shape[0] * self.resize_coef)
        dim = (width, height)
        resized_image = cv2.resize(image,
                                   dim,
                                   interpolation=cv2.INTER_AREA)

        return resized_image

    def modify_image(self, image):
        """
        Modified image. Find edges
        :param image:
        :return: image modified
        """
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Blur the image
        blurred_image = cv2.bilateralFilter(gray_image, 3, 100, 100)  # 3,75,75?
        # Apply Canny edge detector
        canny = cv2.Canny(blurred_image, 35, 400)

        return canny

    def draw_lines(self, lines, image):
        import random, os
        window_name = "Lines detected"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        path = r"D:\Desktop\system_output"

        # while cv2.waitKey(1) < 0:
        #     # Each line is a tuple (coordinates list in a list), length of this line
        #     for line in lines:
        #         cv2.line(image,
        #                  (line[0][0][0], line[0][0][1]),
        #                  (line[0][0][2], line[0][0][3]),
        #                  (0, 0, 255), 1,
        #                  cv2.LINE_AA)
        #
        #
        #     cv2.imwrite(os.path.join(path, str(random.randint(1,10000)) + '.jpg'), image)
        #     cv2.imshow(window_name, image)

        # Each line is a tuple (coordinates list in a list), length of this line
        for line in lines:
            cv2.line(image,
                     (line[0][0][0], line[0][0][1]),
                     (line[0][0][2], line[0][0][3]),
                     (0, 0, 255), 1,
                     cv2.LINE_AA)

        cv2.imwrite(os.path.join(path, str(random.randint(1, 10000)) + '.jpg'), image)


    def extract_lines(self, modified_image):
        """
        Extracts lines longer than a certain threshold in the image provided
        :param image:
        :return:
        """
        lines = cv2.HoughLinesP(image=modified_image,
                                rho=3,  # Distance resolution of accumulator in pixels
                                theta=np.pi/180,  # Angle resolution of the accumulator (radians)
                                threshold=100,  # Accumulator threshold parameter. Only lines that got enough votes get returned
                                lines=np.array([]),  # Output vector of lines. Each line (x1, y1, x2, y2)
                                minLineLength=self.min_line_lenght,  # Min line lenght. Segments shorter are rejected
                                maxLineGap=self.max_line_gap)  # Max allowed gap between points on the same line to link them

        return lines

    def vertical_lines(self, lines):
        """
        Searches for vertical lines (its angle is greater a certain threshold). Calculate
        their length
        :param lines:
        :return: list of tuples
        """
        # We do not want to consider horizontal lines (below the threshold)
        angle_threshold = 60
        # To store all vertical lines
        vertical_lines = list()

        for index, line_coordinates in enumerate(lines):
            x1 = line_coordinates[0][0]
            y1 = line_coordinates[0][1]
            x2 = line_coordinates[0][2]
            y2 = line_coordinates[0][3]
            # Calculate line angle and check if its angle is greater than the threshold
            # to discard horizontal lines
            angle = abs(round(np.rad2deg(np.arctan2((y2 - y1), (x2 - x1))), 2))
            if angle < angle_threshold:
                continue
            # In case multiple parallel lines found, select the longest ones
            line_lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            vertical_lines.append((lines[index], angle, line_lenght))

        return vertical_lines

    def get_N_longest_lines(self, lines):
        # Make sure lines is list
        assert type(lines) == list, "Wrong variable type! Check"
        # Sort the lines based on their length (coordinates, length)
        lines.sort(key = lambda item: item[-1], reverse=True)
        # Take only 6 longest, else return all
        if len(lines) > 6:
            return lines[:6]
        else:
            return lines

    def parallel_lines(self, lines):
        # We're looking for parallel lines. Lines whose angles are within 1 degree range
        # are considered parallel
        # O(n2) complexity. Shame. Very fucking slow my man. Come up with something more efficient.
        max_angle_difference = 0.1
        parallel_lines = list()
        for i in range(len(lines) - 1):
            parallel = list()
            parallel.append(lines[i])

            for j in range(i + 1, len(lines)):
                if abs(lines[i][1] - lines[j][1]) <= max_angle_difference:
                    parallel.append(lines[j])

            if len(parallel) > 1:
                parallel_lines += parallel

        return parallel_lines
