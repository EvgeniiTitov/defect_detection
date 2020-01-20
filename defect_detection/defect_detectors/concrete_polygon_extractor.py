import cv2
import os
import numpy as np

class LineExtender:
    """

    """
    def extend_lines(self,
                     image,
                     the_lines):

        """
        Extends the lines provided within an image (until the lines reach
        image's boundaries)
        :param image: an image on which the lines were detetced
        :param the_lines: list of lists, lines
        :return: coordinates of extended lines, list of lists
        """
        import math

        lines_extended = list()

        for line in the_lines:

            x1, y1 = line[0][0], line[0][1]
            x2, y2 = line[1][0], line[1][1]

            curr_lenght = math.sqrt((x1 - x2)**2 + (y2 - y1)**2)

            # y = 0
            x_top = int(round(x1 + (x1 - x2) / curr_lenght * y1))

            # y = image.shape[0]
            x_bottom = int(
                round(x2 + (x2 - x1) / curr_lenght * (image.shape[0] - y2))
                           )
            # Dots are intentionally appended *flat* to the list, not typical
            # syntax (x1,y1), (x2,y2) etc
            #lines_extended.append([(x_top, 0), (x_bottom, image.shape[0])])
            lines_extended.append([x_top, 0])
            lines_extended.append([x_bottom, image.shape[0]])

        return lines_extended

    def add_secondLine_and_extend(self,
                                  image,
                                  line):
        """
        Draws a line parallel to the one found by the algorithm. We assume the
        pole is in the middle of an image, which is almost often the case because
        the neural network places it in the middle of the bounding box
        :param image:
        :param the_line:
        :return:
        """
        import math

        lines_extended = list()
        # First extend the line detected by the algorithm
        x1, y1 = line[0][0][0], line[0][0][1]
        x2, y2 = line[0][1][0], line[0][1][1]

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


class PolygonRetriever:
    """
    Class tasked with retrieving image section between the lines (pole edges)
    detected in order to calculate pole's inclination angle.
    Since the lines are never full image height, it employs the LineExtender
    class to extend those lines.
    """
    def __init__(
            self,
            line_extender
    ):

        self.line_extender = line_extender

    def retrieve_polygon(
            self,
            image,
            the_lines
    ):
        """

        :param path_to_image:
        :param the_lines:
        :return:
        """

        # Since lines are usually of varying length and almost always are
        # shorter than image's height, extend them first to successfully extract
        # the area confined by them
        extended_lines = list()

        # To check when just one line was detected
        #the_lines = [the_lines[1]]

        if len(the_lines) == 2:
            extended_lines += self.line_extender.extend_lines(image=image,
                                                              the_lines=the_lines)
        else:
            # If the algorithm failed to find two lines and returned only one,
            # we need to draw approximate second line to extract the area in between
            extended_lines += self.line_extender.add_secondLine_and_extend(image=image,
                                                                           line=the_lines)

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
