import cv2
import numpy as np
from neural_networks.detections import DetectedObject

class NeuralNetwork:
    """
    Class accommodating all neural networks involved plus any nn related functions
    """
    conf_threshold = 0.3
    NMS_threshold = 0.2
    input_width = 608  # lower value seems to be speeding up performance
    input_height = 608
    
    def __init__(self):
        # Parse and load configuration data, weights, classes upon class initialization
        self.configuration_2_classes, self.configuration_3_classes, self.block_1_weights,\
        self.block_2_weights_metal, self.block_2_weights_concrete, self.classes_txt = self.load_files()[0]
        self.classes = self.load_files()[1]

        # Initialize neural networks
        self.block_1_NN = self.setup_network(self.configuration_2_classes, self.block_1_weights)
        self.block_2_NN_metal = self.setup_network(self.configuration_2_classes, self.block_2_weights_metal)
        #self.block_2_NN_concrete = self.setup_network(self.configuration_3_classes, self.block_2_weights_concrete)

    def load_files(self):
        """
        Extracts configuration information from the txt files.
        :return: paths to the configuration files
        """
        # ? Make this and setup NN uncallable from the outside
        with open(r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\settings.txt", "rt") as f:
            content = [line.rstrip("\n") for line in f]
        with open(r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\obj.names.txt", "rt") as f:
            classes = [line.rstrip("\n") for line in f]
        
        return (content, classes)

    def setup_network(self, configuration, weights):
        """
        Sets up a neural network using the configurations and weights provided
        :param configuration: YOLO configuration file
        :param weights: YOLO pretrained weights for custom object detection
        :return: neural net initialized and ready for detection
        """
        neural_net = cv2.dnn.readNetFromDarknet(configuration, weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return neural_net

    def create_blob(self, image):
        """
        Creates and returns a blob that gets later fed to a neural network 
        """
        return cv2.dnn.blobFromImage(image,1 / 255, (self.input_width,self.input_height),
                                     [0,0,0], 1, crop=False)
        
    def get_output_names(self, NN):
        """
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106']
        :param NN: neural net
        :return: YOLO outputting layers
        """
        layers_names = NN.getLayerNames()
        
        return [layers_names[i[0]-1] for i in NN.getUnconnectedOutLayers()]

    def widen_bounding_box(self, object):
        '''
        Widens bounding box and returns its new coordinates for METAL poles.
        For instance, clear overlapping here: left, top, right, bottom
        Object detected: [0, 0.6957460641860962, 431, 130, 603, 594]
        344 130 723 594
        Object detected: [0, 0.452914834022522, 675, 357, 768, 571]
        540 357 921 571
        '''
        coef_1 = 0.8
        coef_2 = 1.2
        left_boundary = int(object.BB_left*coef_1)
        right_boundary = int(object.BB_right*coef_2) if int(object.BB_right*coef_2) < \
                             self.image_width else self.image_width

        return (left_boundary, object.BB_top, right_boundary, object.BB_bottom)


    def get_predictions_block1(self, image):
        """
        :param image: image or video frame to perform utility pole detection
        :return: images detected
        """
        # Memorize image's size, will be used in postprocess and for widening BBs
        self.image_height, self.image_width = image.shape[0], image.shape[1]
        # Create a blob from the image
        blob = self.create_blob(image)
        # Pass the image to the neural network
        self.block_1_NN.setInput(blob)
        # Get output YOLO layers
        layers = self.get_output_names(self.block_1_NN)
        # Run forward pass to get output from 3 output layers (BBs). List of 3 numpy
        # matrices of shape (507, 6),(2028, 6),(8112, 6)
        output = self.block_1_NN.forward(layers)
        poles = self.postprocess(image, output)
        # If a pole detected is a metal pole. Widen (probably even heighten) coordinates
        # of this object to address the issue when insulators sticking out horizontally
        # do not get included in the object's bounding box.
        metal_counter, concrete_counter = 1,1
        for pole in poles:
            if pole.class_id == 0: # metal
                # CONDITION TO MAKE SURE NEW BB DO NOT OVERLAP!
                left, top, right, bottom = self.widen_bounding_box(pole)
                pole.update_object_coordinates(left, top, right, bottom)
                # Dynamically specify what object it is just to ease my life
                pole.object_class = "metal_{}".format(metal_counter)
                metal_counter += 1
            else:
                # ! SQUEZZE COORDINATES FOR POLE DETECTION ON CONCRETE POLES
                pole.object_class = "concrete_{}".format(concrete_counter)
                concrete_counter += 1
        
        return poles
    
    def get_predictions_block2_metal(self, image):
        """
        Detects utility pole components
        :param image: 
        :return: 
        """
        self.image_height, self.image_width = image.shape[0], image.shape[1]
        blob = self.create_blob(image)
        self.block_2_NN_metal.setInput(blob)
        layers = self.get_output_names(self.block_2_NN_metal)
        output = self.block_2_NN_metal.forward(layers)
        components = self.postprocess(image, output)
        
        return components

    def get_predictions_block2_concrete(self):
        pass
    
    def postprocess(self, frame, outs):
        '''
        Processes data outputted from 3 YOLO layers. Removes BBs with low confidence using
        non-max suppression. 
        '''
        class_ids, confidences, boxes = [],[],[]
        # Check all detections from 3 YOLO layers. Discard bad ones.
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0]*self.image_width)
                    center_y = int(detection[1]*self.image_height)
                    width = int(detection[2]*self.image_width)
                    height = int(detection[3]*self.image_height)
                    left = abs(int(center_x - width / 2))
                    top = abs(int(center_y - height / 2))

                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left,top,width,height])
        # Perform non-max suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.NMS_threshold)
        
        objects_detected = list()
        for index, i in enumerate(indices):
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # For convenience each object detected gets represented as a class object
            object = DetectedObject(class_ids[i],confidences[i], left, top, left+width, top+height)
            objects_detected.append(object)
            
        return objects_detected
    