# python C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection --image=D:\Desktop\Test_Dir\00110.jpg

#from defect_detection.preprocessing.preprocessing import *
from neural_networks.neural_network import NeuralNetwork
from neural_networks.detections import DetectionImageSection
import cv2
import os
import sys
import argparse
import time
import collections
import numpy as np

parser = argparse.ArgumentParser(description='Defect detection with YOLO and OpenCV')
parser.add_argument('--image', help='Path to an image.')
parser.add_argument('--video', help='Path to a video.')
parser.add_argument('--folder', help='Path to a folder containing images to get processed')
parser.add_argument('--crop', default=False, help='Crop out and save objects detected')
parser.add_argument('--save_path', default=r'D:\Desktop\system_output', help="Path to where save images afterwards")
arguments = parser.parse_args()


def draw_bounding_boxes(objects_detected, frame):
    """
    Functions that draws bb around the objects found by the neural nets
    :param objects_detected:
    :return:
    """
    line_thickness = (frame.shape[0] * frame.shape[1] // 1_000_000)
    text_size = 0.5 + (frame.shape[0] * frame.shape[1] // 5_000_000)
    text_boldness = 1 + (frame.shape[0] * frame.shape[1] // 2_000_000)
    for image_section, objects in objects_detected.items():
        for element in objects:
            cv2.rectangle(frame, (image_section.left+element.left, image_section.top+element.top),
                                 (image_section.left+element.right, image_section.top+element.bottom),
                                 (0, 255, 0), line_thickness)
            label = "{}:{:1.2f}".format(element.object_class[:-4], element.confidence)
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
            top = max(element.top + image_section.top, label_size[1])
            cv2.putText(frame, label, (element.left + image_section.left, top),
                        cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_boldness)


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
                cv2.imwrite(os.path.join(crop_path, frame_name), cropped_frame)
            else:
                cropped_frame = image_section.frame[element.top+image_section.top:element.bottom+image_section.top,
                                                    element.left+image_section.left:element.right+image_section.left]
                image_name = element.object_class + '_' + os.path.split(path_to_image)[-1]
                cv2.imwrite(os.path.join(crop_path, image_name), cropped_frame)


def object_detection(image=None, cap=None, video_writer=None):
    """
    Performs object detection and its postprocessing on images or video frames. On each frame
    or one in N frames! (can we keep bb drawn for the frames we just show without detecting?)
    :param image:
    :param cap: When working with video input
    :param video_writer: When working with video input
    """
    global frame_counter
    # Class accommodating all neural networks and associated functions
    NN = NeuralNetwork()
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
        whole_image = DetectionImageSection(frame, "image, utility poles")
        # Run neural net and see if we got any poles detected
        poles_detected = NN.predict_poles(frame)
        # Save poles detected along side the image section in which detection took place
        if poles_detected:
            for pole in poles_detected:
                objects_detected[whole_image].append(pole)

        # BLOCK 2
        # If no poles found, check for close-up components. Send the whole image
        # to the second set of NNs
        components_detected = list()
        if poles_detected:
            # Separately send each pole detected to components detecting neural net
            for pole in poles_detected:
                # Get new image section - pole's coordinates.
                pole_subimage = np.array(frame[pole.top:pole.bottom, pole.left:pole.right])
                pole_section = DetectionImageSection(pole_subimage, "subimage, components")
                # Above we save subimage size. Now save its coordinates relatively to the original image!
                pole_section.save_relative_coordinates(pole.top, pole.left, pole.right, pole.bottom)

                # For different pole classes, we will be using different weights.
                # FOR NOW WE USE THE SAME WEIGHT, SAME NN! GET NEW WEIGHTS FOR 3 CLASSES
                if pole.class_id == 0:  # metal
                    components_detected += NN.predict_components_metal(pole_subimage)
                elif pole.class_id == 1:  # concrete
                    components_detected += NN.predict_components_metal(pole_subimage)

                # Check if any components have been found. Save to the dictionary
                if components_detected:
                    for component in components_detected:
                        objects_detected[pole_section].append(component)
        else:
            # No poles detected, send the whole frame to check for close-up components
            components_detected += NN.predict_components_metal(frame)
            whole_image = DetectionImageSection(frame, "image, components")
            if components_detected:
                for component in components_detected:
                    objects_detected[whole_image].append(component)

        # BLOCK 3
        # DEFECT DETECTION ON THE OBJECTS DETECTED
        pass

        # SAVE DEFECTS IF FOUND TO A DATABASE
        pass

        # SAVE RESULTS IF REQUIRED, DRAW BOUNDING BOXES
        if arguments.crop:
            save_objects_detected(objects_detected)
        draw_bounding_boxes(objects_detected, frame)

        # Save a frame/image with BBs drawn on it
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

        # ! BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED (FILTERS)
        
        object_detection(image=image)

    elif arguments.folder:
        if not os.path.isdir(arguments.folder):
            print("The provided file is not a folder")
            sys.exit()
        for image in os.listdir(arguments.folder):
            if not any(image.endswith(ext) for ext in [".jpg", ".JPG", ".jpeg", ".JPEG"]):
                continue
            path_to_image = os.path.join(arguments.folder, image)
            image = cv2.imread(path_to_image)
                               
            # ! BEFORE IMAGE MIGHT NEEDS TO BE PREPROCESSED 
            
            object_detection(image=image)

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
        object_detection(cap=cap, video_writer=video_writer)

    print("All input has been processed")
    sys.exit()


if __name__ == "__main__":
    main()
