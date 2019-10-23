import cv2, os, sys
import numpy as np


class TiltChecker:
    def __init__(self,
                 min_line_lenght=100,
                 max_line_gap=100,
                 tilt_threshold=3,
                 resize_coef=1):
        self.min_line_lenght = min_line_lenght
        self.max_line_gap = max_line_gap
        self.tilt_thresh = tilt_threshold
        self.resize_coef = resize_coef

    def check_pole(self, image, pitch=0, roll=0):
        """
        :param image: numpy array. Image section containing a pole cropped
        :param pitch: value of the pitch angle
        :param roll: value of the roll angle
        :return: angle of inclination, pole status (green, yellow, red)
        """
        # Resize image
        resized = self.resize_image(image)
        # Modify the image (filters, resizing), find edges
        modified_image = self.modify_image(resized)
        # Find all lines on the image
        lines = self.extract_lines(modified_image)

        # -----------------DIFFERENT APPROACHES TO PARSE THE LINES-------------------

        # Find the coordinates of the longest line
        #the_line = self.find_longest_line(lines)[0]  # [0] since list in the list gets returned

        # Find the most vertical line (to prevent finding wires)
        #the_line = self.find_most_vertical_line(lines)[0]

        # Combination of the two
        if lines is None:
            print("No lines found")
            return
        # Find N (5) longest lines
        longest_lines = self.N_longest_lines(lines)
        # Among those 5 lines find the most vertical line - the line
        the_line = self.find_most_vertical_line(longest_lines)[0]
        # Calculate angle between the line and the bottom edge of an image
        angle_rel_to_horizon = self.calculate_angle(resized, the_line)
        # Convert the calculations to be relatively to the vertical line
        angle = 90 - angle_rel_to_horizon
        print("angle:", angle)

        # Error management
        if pitch > 0:
            angle -= pitch
        else:
            angle += pitch

        # Green, yellow, red
        pass

        print("angle with error:", angle)
        # HERE WE HAVE TO CONSIDER THE ERROR PITCH, ROLL
        # THEN ANNOUCE IF POLE ON THE PICTURE IS DEFECTED

        # FOR TESTING PURPOSES
        self._draw_line(resized, the_line)

    def calculate_angle(self, image, line):
        """
        Calculate angle between the line found and image's bot
        :param image:
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
        Resized image
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

    def find_longest_line(self, lines):
        """
        NOTE! TIME COMPLEXITY O(n)!
        Calculates and returns coordinates of the longest line
        :param lines:
        :return:
        """
        longest_line = (0,0)  # index of the line, its lenght
        for index, line_coordinates in enumerate(lines):
            # Extra index since its a list in the list
            x1 = line_coordinates[0][0]
            y1 = line_coordinates[0][1]
            x2 = line_coordinates[0][2]
            y2 = line_coordinates[0][3]
            line_lenght = ((x2-x1)**2 + (y2-y1)**2)**0.5
            if line_lenght > longest_line[-1]:
                longest_line = (index, line_lenght)

        return lines[longest_line[0]]  # Index of the longest line

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
        # Collect indices from the list above
        indices_of_longest = [i for i,lenght in longest_lines]

        # Return 5 longest lines
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

    def _draw_line(self, image, line):
        #print(line)
        cv2.line(image,
                (line[0], line[1]),
                (line[2], line[3]),
                (0, 0, 255), 2,
                 cv2.LINE_AA)

        window_name = "Lines"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while cv2.waitKey(1) < 0:
            cv2.imshow(window_name, image)
            #cv2.imwrite(os.path.join(save_path, item), image)

if __name__ == "__main__":
    # PROCESS A SINGLE IMAGE
    path = r"D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\pole_tilt_test\crop_image\DJI_0405.JPG"
    # path = r"D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\pole_tilt_test\my_tests\cropped\26.JPG"
    image = cv2.imread(path)
    checker = TiltChecker(min_line_lenght=100,
                          max_line_gap=200,
                          resize_coef=0.33)
    checker.check_pole(image,2,0)

    # # PROCESS ALL IMAGES IN A FOLDER
    #folder = r"D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\pole_tilt_test\my_tests\cropped"
    # folder = r"D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\pole_tilt_test\crop_image"
    # save_path = r"D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\pole_tilt_test\my_tests\cropped\blur_150"
    # # line lenght = 100, line gap 200 so far the best result
    # checker = TiltChecker(min_line_lenght=100,
    #                       max_line_gap=200)
    #
    # for item in os.listdir(folder)[:1]:
    #     path_to_item = os.path.join(folder, item)
    #     if os.path.isdir(path_to_item):
    #         continue
    #     print("\nProcessing:", item)
    #     try:
    #         image = cv2.imread(path_to_item)
    #         checker.check_pole(image)
    #     except:
    #         print("Failed on:", item)
    #         continue
