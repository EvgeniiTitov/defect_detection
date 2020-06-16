import numpy as np
import cv2


class WoodCracksDetector:

    @staticmethod
    def detect_cracks(image: np.ndarray) -> np.ndarray:
        #image = WoodCracksDetector.resize_image(image, height=300)
        image_original = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # was (5.5)
        close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
        close = cv2.absdiff(close, image)
        thresh = cv2.threshold(image, 244, 255, cv2.THRESH_BINARY_INV)[1]
        thin_thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, se, iterations=3)
        cv2.normalize(close, close, 0, 255, cv2.NORM_MINMAX)
        crack = cv2.bitwise_and(close, thin_thresh)
        crack = cv2.bitwise_not(crack)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        canny = cv2.Canny(crack, 100, 255, 1)
        dilate = cv2.dilate(canny, kernel, iterations=2)
        cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        min_area = 200
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area:
                cv2.drawContours(image_original, [c], -1, (36, 255, 12), 2)

        return image_original

    @staticmethod
    def resize_image(
            image: np.ndarray,
            width: int = None,
            height: int = None,
            inter=cv2.INTER_AREA
    ) -> np.ndarray:
        if width is None and height is None:
            return image

        h, w = image.shape[:2]
        if width is None:
            scaling_factor = height / float(h)
            dim = (int(w * scaling_factor), height)
        else:
            scaling_factor = width / float(w)
            dim = (width, int(scaling_factor * h))

        return cv2.resize(image, dim, interpolation=inter)
