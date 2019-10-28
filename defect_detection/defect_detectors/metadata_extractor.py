import re
import os


class MetaDataExtractor:
    def __init__(self, kernel_size=9):
        self.kernel_size = kernel_size

    def get_error_values(self, path_to_image):
        """
        Extracts metadata we are interested in
        :return:
        """
        with open(path_to_image, encoding="utf8", errors="ignore") as d:
            metadata = d.read()
            if not metadata:
                print(f"The image {os.path.split(path_to_image)[-1]} "
                      f"provided does not have any metadata associated with it")
                return
            start = metadata.find("<x:xmpmeta")
            end = metadata.find("</x:xmpmeta")
            data = metadata[start : end + 1]

            return self.calculate_error(data)

    def calculate_error(self, metadata):
        """
        Calculates orientation errors (pitch, roll) based on the metadata extracted
        :return: pitch, roll values in tuple
        """
        pitch = str(re.findall(r"drone-dji:FlightPitchDegree=\D\+?\-?\d+\.\d+\D", metadata))
        roll = str(re.findall(r"drone-dji:GimbalRollDegree=\D\+?\-?\d+\.\d+\D", metadata))

        pitch_angle = re.findall(r"\d+.\d+", pitch)
        roll_degree = re.findall(r"\d+.\d+", roll)

        if any((pitch_angle, roll_degree)):
            # Since values are stored in lists
            return float(pitch_angle[0]), float(roll_degree[0])
        else:
            return None
