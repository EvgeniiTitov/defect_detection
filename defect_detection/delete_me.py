from defect_detection.preprocessing import MetaDataExtractor
from defect_detection.defect_detectors.pole_tilt_checker import TiltChecker
import cv2, sys


if __name__ == "__main__":
    path = r"D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\pole_tilt_test\crop_image\DJI_0527.JPG"
    image = cv2.imread(path)

    preprocessor = MetaDataExtractor()
    pitch, roll = preprocessor.get_error_values(path)
    print("Meta values:", pitch, roll)
    detector = TiltChecker(min_line_lenght=100,
                          max_line_gap=200,
                          resize_coef=0.2)
    detector.check_pole(image, pitch, roll)
