# python C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection

from defect_detection.preprocessing import preprocessing
from defect_detection.neural_networks.neural_networks import Neural_Network
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

def detecting_on_images(image, NN):
    '''
    :param image: numpy array - image getting processed 
    :param NN: 
    :return: 
    '''
    poles_detected = NN.get_predictions_block_1_matrix(image)
    

def detecting_on_video():
    '''
    
    :return: 
    '''
    pass

def main():
    global save_path, crop_path
    # Check if information (images, video) has been provided
    if not any((arguments.image, arguments.video, arguments.folder)):
        print("You have not provided a single source of data. Try again")
        sys.exit()
    # Class accomodating all neural networks and associated functions
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
        detecting_on_images(image, NN)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()

        for image in os.listdir(arguments.folder):
            if not any(image.endswith(extension) for extension in [".jpg",".JPG"]):
                continue
            # BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED (FILTERS)
            image = cv2.imread(arguments.image)
            detecting_on_images(image, NN)

        print("All images in the folder have been processed")
        sys.exit()

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

if __name__=="__main__":
    main()