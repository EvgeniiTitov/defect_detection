import os
import sys
import argparse
import time
import cv2
from neural_networks.predictors import ResultsHandler, PoleDetector, ComponentsDetector, DefectDetector


def parse_args():
    parser = argparse.ArgumentParser()
    # Type of input data
    parser.add_argument('--image', type=str,  help='Path to an image.')
    parser.add_argument('--video', type=str, help='Path to a video.')
    parser.add_argument('--folder', type=str, help='Path to a folder containing images.')
    # Managing results
    parser.add_argument('--crop_path', type=str, default=False,
                        help='Path to crop out and save objects detected')
    parser.add_argument('--save_path', type=str, default=r'D:\Desktop\system_output',
                        help="Path to where save images afterwards")
    arguments = parser.parse_args()

    return arguments


def detection(save_path,
              path_to_input,
              cap=None,
              image=None,
              crop_path=None,
              video_writer=None):

    frame_counter = 0
    input_type = True
    objects = dict()
    pole_detector = PoleDetector()
    component_detector = ComponentsDetector()
    defect_detector = DefectDetector()

    while cv2.waitKey(1) < 0:
        start_time = time.time()
        if all((cap, video_writer)):
            input_type = False
            has_frame, frame = cap.read()
            if not has_frame:
                sys.exit()
        else:
            frame = image

        handler = ResultsHandler(image=frame,
                                 path_to_image=path_to_input,
                                 save_path=save_path,
                                 cropped_path=crop_path,
                                 input_photo=input_type,
                                 video_writer = video_writer,
                                 frame_counter = frame_counter)

        # BLOCK 1,2. Detect poles and elements on them
        poles = pole_detector.predict(frame)
        components = component_detector.predict(frame, poles)
        # Merge all objects found into one dictionary
        for dictionary in (poles, components):
            objects.update(dictionary)

        # BLOCK 3. TBC. SEND ALL OBJECTS TO DEFECT DETECTOR.
        # defect_detector.find_defects(objects)

        # SAVE DEFECTS/OBJECTS TO A DATABASE
        if crop_path:
            handler.save_objects_detected(objects)
        handler.draw_bounding_boxes(objects)
        handler.save_frame()


        cv2.imshow(window_name, frame)
        frame_counter += 1
        end_time = time.time()
        print("FPS:", end_time - start_time)

        # In case of image break out of WHILE loop. Show image for N sec.
        if len(image) > 0:
            cv2.waitKey(2000)
            return


if __name__ == "__main__":
    arguments = parse_args()

    if not any((arguments.image, arguments.video, arguments.folder)):
        print("You have not provided a single source of data. Try again")
        sys.exit()

    window_name = "Defect detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    save_path = arguments.save_path

    crop_path = None
    if arguments.crop_path:
        crop_path = arguments.crop

    if arguments.image:
        if not os.path.isfile(arguments.image):
            print("The provided file is not an image")
            sys.exit()
        image = cv2.imread(arguments.image)
        detection(save_path=save_path,
                  path_to_input=arguments.image,
                  crop_path=crop_path,
                  image=image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder. Double check!")
            sys.exit()
        for image in os.listdir(arguments.folder):
            if not any(image.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                continue
            print("\nProcessing:", image)
            path_to_image = os.path.join(arguments.folder, image)
            image = cv2.imread(path_to_image)
            detection(save_path=save_path,
                      path_to_input=path_to_image,
                      crop_path=crop_path,
                      image=image)

    elif arguments.video:
        if not os.path.isfile(arguments.video):
            print("The provided file is not a video")
            sys.exit()
        cap = cv2.VideoCapture(arguments.video)
        video_name = os.path.split(arguments.video)[-1][:-4]
        output_file = video_name + "_OUT.avi"
        video_writer = cv2.VideoWriter(output_file,
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                       (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        detection(save_path=save_path,
                  path_to_input=arguments.video,
                  crop_path=crop_path,
                  cap=cap,
                  video_writer=video_writer)
