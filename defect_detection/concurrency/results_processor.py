import threading
import cv2
import os
import numpy as np

class ResultsProcessor(threading.Thread):
    """
    Draws BBs
    Writes text
    Constructs JSON if required
    """
    def __init__(
            self,
            save_path,
            queue_from_defect_detector,
            input_type,
            filename,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.done = False

        self.Q_in = queue_from_defect_detector
        self.save_path = save_path

        self.input_type = input_type
        self.filename = filename

    def run(self) -> None:

        video_writer = None

        while not self.done:

            item = self.Q_in.get(block=True)

            if item == "END":
                break

            image, defects, detected_objects = item

            if video_writer is None and self.input_type == "video":
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(os.path.join(self.save_path, self.filename + '_out.avi'),
                                               fourcc, 30,
                                               (image.shape[1], image.shape[0]), True)

            # Draw BBs
            self.draw_bbs(detected_objects=detected_objects, image=image)

            # Save frame
            self.save_frame(image=image, video_writer=video_writer)

            # TODO: Generate JSON if an image OR end of the video. WE NEED TO CHECK
            # TODO: IF "END" has come, then generate and output the result

        return

    def draw_bbs(self, detected_objects, image):

        # TODO: Somehow need to get an frame to draw BB on it

        colour = (0, 255, 0)

        for subimage, elements in detected_objects.items():
            for element in elements:

                if element.object_name == "insl":
                    colour = (210, 0, 210)
                elif element.object_name == "dump":
                    colour = (255, 0, 0)
                elif element.object_name == "pillar":
                    colour = (0, 128, 255)

                    if element.inclination:
                        text = f"Angle: {element.inclination}"
                        cv2.putText(image, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)

                cv2.rectangle(image, (subimage.left + element.BB_left, subimage.top + element.BB_top),
                              (subimage.left + element.BB_right, subimage.top + element.BB_bottom),
                              colour, self.line_text_size(image)[0])

                label = "{}:{:1.2f}".format(element.object_name, element.confidence)

                label_size, base_line = cv2.getTextSize(label,
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        self.line_text_size(image)[1], 1)

                top = max(element.top + subimage.top, label_size[1])

                cv2.putText(image, label,
                            (element.left + subimage.left, top),
                            cv2.FONT_HERSHEY_SIMPLEX, self.line_text_size(image)[1],
                            (0, 0, 0), self.line_text_size(image)[-1])

    def save_frame(
            self,
            image: np.ndarray,
            video_writer
    ):
        """
        Saves a frame with all BBs drawn on it
        :return:
        """
        if self.input_type == "image":
            image_name = self.filename + "_out.jpg"
            cv2.imwrite(os.path.join(self.save_path, image_name), image)
        else:
            video_writer.write(image.astype(np.uint8))

    def line_text_size(
            self,
            image
    ):
        """
        Method determining BB line thickness and text size based on the original image's size
        :return:
        """
        line_thickness = int(image.shape[0] * image.shape[1] // 1_000_000)
        text_size = 0.5 + (image.shape[0] * image.shape[1] // 5_000_000)
        text_boldness = 1 + (image.shape[0] * image.shape[1] // 2_000_000)

        return line_thickness, text_size, text_boldness