# python C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection

#from defect_detection.preprocessing.preprocessing import *
from neural_networks.neural_network import Neural_Network
import cv2
import os
import sys
import argparse
import time

parser = argparse.ArgumentParser(description='Defect detection with YOLO and OpenCV')
parser.add_argument('--image', help='Path to an image file.')
parser.add_argument('--video', help='Path to a video file.')
parser.add_argument('--folder', help='Path to a folder containing images to get processed')
parser.add_argument('--crop', default=False, help='Crop out and save objects detected')
parser.add_argument('--save_path', default=r'\detection_outputs', help="Path to where save images afterwards")
arguments = parser.parse_args()

def object_detection(NN, image=None, cap=None, video_writer=None):
    """
    Performs object detection and its postprocessing
    :param NN: neural network class featuring all neural networks and associated functions
    :param image:
    :param cap: When working with video input
    :param video_writer: When working with video input
    """
    frame_counter = 0
    # For images the loop gets executed once, for videos till they last
    while cv2.waitKey(1) < 0:
        start_time = time.time()
        frame = None
        # Process input: a single image or a cap - video object
        if all((cap, video_writer)):
            has_frame, frame = cap.read()
            if not has_frame:
                cv2.waitKey(2000)
                return
            # In case we want to process one in N frames to speed up performance
            # if not frame_counter % 30 == 0:
            #     cv2.imshow(window_name, frame)
            #     frame_counter += 1
            #     continue
        elif len(image) > 0:
            frame = image

        # BLOCK 1
        poles_detected = NN.get_predictions_block1(frame)

        if not poles_detected:
            # If no poles found, check for close-up components
            pass





        frame_counter += 1
        end_time = time.time()
        print("FPS: ", end_time - start_time)

        if len(image) > 0:
            return

def main():
    global save_path, crop_path, window_name
    # Check if information (images, video) has been provided
    if not any((arguments.image, arguments.video, arguments.folder)):
        print("You have not provided a single source of data. Try again")
        sys.exit()
    # Class accommodating all neural networks and associated functions
    NN = Neural_Network()
    # Create window to display images/video frames
    window_name = "Defect detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # PARSE USER INPUT
    save_path = arguments.save_path
    # Check if user wants to crop out the objects detected
    crop_path = None
    if arguments.crop:
        crop_path = arguments.crop

    # FOR IMAGES HOW ARE WE GOING TO MAKE SURE WE WORK WITH jpg or JPG ONL?
    if arguments.image:
        if not os.path.isfile(arguments.image):
            print("The provided file is not an image")
            sys.exit()
        # BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED (FILTERS)
        image = cv2.imread(arguments.image)
        object_detection(NN, image=image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()
        for image in os.listdir(arguments.folder):
            if not any(image.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                continue
            path_to_image = os.path.join(arguments.folder, image)
            # BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED (FILTERS)
            image = cv2.imread(path_to_image)
            object_detection(NN, image=image)

    elif arguments.video:
        if not os.path.isfile(arguments.video):
            print("The provided file is not a video")
            sys.exit()
        cap = cv2.VideoCapture(arguments.video)
        video_name = os.path.split(arguments.video)[-1][:-4]
        output_file = video_name + "_out.avi"
        video_writer = cv2.VideoWriter(output_file,
                                       cv2.VideoWriter_fourcc('M','J','P','G'), 10,
                                       (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        object_detection(NN, cap=cap, video_writer=video_writer)

    print("All input has been processed")
    sys.exit()

if __name__ == "__main__":
    main()
