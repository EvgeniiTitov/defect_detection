import cv2
import numpy as np


class TiltChecker:
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
