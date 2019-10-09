# python C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection

#from defect_detection.preprocessing.preprocessing import *
from neural_networks.neural_network import NeuralNetwork
from neural_networks.detections import DetectionSection
import cv2
import os
import sys
import argparse
import time
import collections
import numpy as np

parser = argparse.ArgumentParser(description='Defect detection with YOLO and OpenCV')
parser.add_argument('--image', help='Path to an image file.')
parser.add_argument('--video', help='Path to a video file.')
parser.add_argument('--folder', help='Path to a folder containing images to get processed')
parser.add_argument('--crop', default=False, help='Crop out and save objects detected')
parser.add_argument('--save_path', default=r'C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\detection_outputs', help="Path to where save images afterwards")
arguments = parser.parse_args()


def draw_bounding_boxes(objects_detected):
    """
    Functions that draws bb around the objects found by the neural nets
    :param objects_detected:
    :return:
    """
    original_image = frame  # frame made global
    for image_section, objects in objects_detected.items():
        for element in objects:
            cv2.rectangle(original_image, (image_section.left+element.left, image_section.top+element.top),
                                          (image_section.left+element.right, image_section.top+element.bottom),
                                          (0, 255, 0), 3)
            label = "{}:{:1.2f}".format(element.object_class[:-4], element.confidence)
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            top = max(element.top + image_section.top, label_size[1])
            cv2.putText(original_image, label, (element.left + image_section.left, top),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)


def save_objects_detected(objects_detected):
    """
    Function that saves objects detected by the neural networks on the disk
    :param objects_detected: Dictionary with objects found
    :return: None
    """
    for image_section, objects in objects_detected.items():
        for element in objects:
            if arguments.video:
                cropped_frame = image_section.frame[element.top+image_section.top:element.bottom+image_section.top,
                                                    element.left+image_section.left:element.right+image_section.left]
                frame_name = "frame" + str(frame_counter) + ".jpg"
                cv2.imwrite(os.path.join(save_path, frame_name), cropped_frame)
            else:
                cropped_frame = image_section.frame[element.top+image_section.top:element.bottom+image_section.top,
                                                    element.left+image_section.left:element.right+image_section.left]
                image_name = element.object_class + '_' + os.path.split(path_to_image)[-1]
                cv2.imwrite(os.path.join(save_path, image_name), cropped_frame)

def object_detection(NN, image=None, cap=None, video_writer=None):
    """
    Performs object detection and its postprocessing on images or video frames. On each frame
    or one in N frames! (can we keep bb drawn for the frames we just show without detecting?)
    :param NN: neural network class featuring all neural networks and associated functions
    :param image:
    :param cap: When working with video input
    :param video_writer: When working with video input
    """
    global frame_counter, frame
    frame_counter = 0
    # For images the loop gets executed once, for videos till they last (have frames)
    while cv2.waitKey(1) < 0:
        start_time = time.time()
        # To keep track of all objects detected on a frame we use dictionary
        # where keys - image section on which detection was performed, values -
        # objects detected there (image:[pole1,pole2], pole1_coordinates:[insulator,dumper])
        objects_detected = collections.defaultdict(list)
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
        # First we consider the whole image. Create an instance reflecting this:
        whole_image = DetectionSection(frame, "utility_poles")
        # Run neural net and see if we got any poles detected
        poles_detected = NN.get_predictions_block1(frame)
        # Save poles detected along side the image section in which detection took place
        if poles_detected:
            for pole in poles_detected:
                objects_detected[whole_image].append(pole)

        # BLOCK 2
        # If no poles found, check for close-up components. Send the whole image
        # to the second set of NNs
        components_detected = list()
        if objects_detected:
            # Separately send each pole detected to components detecting neural net
            for pole in poles_detected:
                # Get new image section - pole's coordinates. 
                pole_image_section = np.array(frame[pole.top:pole.bottom, pole.left:pole.right])
                pole_section = DetectionSection(pole_image_section, "components") # second parameter is name
                # Above we save the cropped frame, its size. Here, we save its coordinates relatively to the original image!
                pole_section.save_relative_coordinates(pole.top, pole.left, pole.right, pole.bottom)
                # For different pole classes, we will be using different weights.
                # FOR NOW WE USE THE SAME WEIGHT, SAME NN! GET NEW WEIGHTS FOR 3 CLASSES
                if pole.class_id == 0:  # metal
                    components_detected += NN.get_predictions_block2_metal(pole_image_section) 
                elif pole.class_id == 1:  # concrete
                    components_detected += NN.get_predictions_block2_metal(pole_image_section)

                # Check if any components have been found. Save to the dictionary
                if components_detected:
                    for component in components_detected:
                        objects_detected[pole_section].append(component)
        else:
            # No poles detected, send the whole frame to check for close-up instances of components
            components_detected += NN.get_predictions_block2_metal(frame)
            whole_image = DetectionSection(frame, "components_close_up")
            if components_detected:
                for component in components_detected:
                    objects_detected[whole_image].append(component)

        # BLOCK 3
        # DEFECT DETECTION ON THE OBJECTS DETECTED
        pass

        # SAVE DEFECTS IF FOUND TO A DATABASE
        pass

        # SAVE RESULTS, DRAW BOUNDING BOXES
        #save_objects_detected(objects_detected)
        draw_bounding_boxes(objects_detected)

        # Save a frame with BBs drawn on it
        if video_writer:
            video_writer.write(frame.astype(np.uint8))
        elif any((arguments.image, arguments.folder)):
            image_name = 'out_' + os.path.split(path_to_image)[-1]
            cv2.imwrite(os.path.join(save_path, image_name), frame)

        cv2.imshow(window_name, frame)

        frame_counter += 1
        end_time = time.time()
        print("FPS:", end_time - start_time)

        if len(image) > 0:
            cv2.waitKey(3000)
            return


def main():
    global save_path, crop_path, window_name, path_to_image
    # Check if information (images, video) has been provided
    if not any((arguments.image, arguments.video, arguments.folder)):
        print("You have not provided a single source of data. Try again")
        sys.exit()
    # Class accommodating all neural networks and associated functions
    NN = NeuralNetwork()
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
        path_to_image = arguments.image
        image = cv2.imread(path_to_image)
        
        # BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED (FILTERS)
        
        object_detection(NN, image=image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()
        for image in os.listdir(arguments.folder):
            if not any(image.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                continue
            path_to_image = os.path.join(arguments.folder, image)
            image = cv2.imread(path_to_image)
                               
            # BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED 
            
            object_detection(NN, image=image)

    elif arguments.video:
        if not os.path.isfile(arguments.video):
            print("The provided file is not a video")
            sys.exit()
        cap = cv2.VideoCapture(arguments.video)
        video_name = os.path.split(arguments.video)[-1][:-4]
        output_file = video_name + "_out.avi"
        video_writer = cv2.VideoWriter(output_file,
                                       cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                       (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        object_detection(NN, cap=cap, video_writer=video_writer)

    print("All input has been processed")
    sys.exit()


if __name__ == "__main__":
    main()
