import cv2
import numpy as np
from neural_networks.detections import DetectedObject

class NeuralNetwork:
    """
    Class accommodating all neural networks involved plus any nn related functions
    """
    conf_threshold = 0.2
    NMS_threshold = 0.2
    input_width = 416  # lower value seems to be speeding up performance
    input_height = 416  # 320, 416, 512,
    
    def __init__(self):
        # Parse and load configuration data, weights, classes upon class initialization
        self.config_2_classes, self.config_3_classes, self.pole_weights, \
        self.components_weights_metal, self.components_weights_concrete, \
        self.classes_txt = self.load_files()[0]
        self.classes = self.load_files()[1]

        # Initialize neural networks
        self.poles_NN = self.setup_network(self.config_2_classes, self.pole_weights)
        self.components_NN_metal = self.setup_network(self.config_2_classes, self.components_weights_metal)
        #self.components_NN_concrete = self.setup_network(self.configuration_3_classes, self.block_2_weights_concrete)

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
        return cv2.dnn.blobFromImage(image, 1 / 255, (self.input_width, self.input_height),
                                     [0, 0, 0], 1, crop=False)
        
    def get_output_names(self, NN):
        """
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106']
        :param NN: neural net
        :return: YOLO outputting layers
        """
        layers_names = NN.getLayerNames()
        
        return [layers_names[i[0]-1] for i in NN.getUnconnectedOutLayers()]

    def enlarge_bounding_box(self, element, image_width, image_height):
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
        # Set new left boundary, move it 20% to the left
        left_boundary = int(element.BB_left * coef_1)
        # Set new right boundary, move it 20% to the right if it doesn't go beyond the edge
        right_boundary = int(element.BB_right * coef_2) if int(element.BB_right * coef_2) < \
                                                           image_width else image_width

        return (left_boundary, element.BB_top, right_boundary, element.BB_bottom)

    def predict_poles(self, image):
        """
        :param image: image or video frame to perform utility pole detection and classification
        :return: images detected
        """
        # Memorize image's size, will be used in postprocess and for widening BBs
        image_height, image_width = image.shape[0], image.shape[1]
        # Create a blob from the image
        blob = self.create_blob(image)
        # Pass the image to the neural network
        self.poles_NN.setInput(blob)
        # Get output YOLO layers
        layers = self.get_output_names(self.poles_NN)
        # Run forward pass to get output from 3 output layers (BBs). List of 3 numpy matrices of shape
        output = self.poles_NN.forward(layers)
        poles = self.postprocess(image, output)

        # If a pole detected is a metal pole. Widen (probably even heighten) coordinates
        # of this object to address the issue when insulators sticking out horizontally
        # do not get included in the object's bounding box.
        metal_counter, concrete_counter = 1, 1
        for pole in poles:
            if pole.class_id == 0: # metal
                
                # ! CONDITION TO MAKE SURE NEW BB DO NOT OVERLAP
                
                # get new coordinates for the left and right boundaries
                left, top, right, bottom = self.enlarge_bounding_box(pole, image_width, image_height)
                # updates object's left and right boundaries
                pole.update_object_coordinates(left, top, right, bottom)
                # Dynamically specify what object it is just to ease my life
                pole.object_class = "metal_{}".format(metal_counter)
                metal_counter += 1
            else:
                
                # ! SQUEEZE COORDINATES FOR POLE DETECTION ON CONCRETE POLES.
                
                # ! MIGHT NEED TO WIDEN BBs FOR CONCRETE AS WELL TO MAKE SURE DUMPERS GET INCLUDED
                
                pole.object_class = "concrete_{}".format(concrete_counter)
                concrete_counter += 1

        return poles
    
    def predict_components_metal(self, image):
        """
        Detects utility pole components
        :param image: 
        :return: 
        """
        self.image_height, self.image_width = image.shape[0], image.shape[1]
        blob = self.create_blob(image)
        self.components_NN_metal.setInput(blob)
        layers = self.get_output_names(self.components_NN_metal)
        output = self.components_NN_metal.forward(layers)
        # Get predictions
        components = self.postprocess(image, output)
        # Perform results modification
        insulator_counter, dumper_counter = 1, 1
        # Perform modifications to the objects found
        
        # ! MIGHT BE A GOOD IDEA TO WIDEN AND HEIGHTEN BBs FOR ELEMENTS
        # SINCE IT OFTEN DOES NOT CONTAIN THE WHOLE OBJECT
        
        for component in components:
            if component.class_id == 0:  # insulator?
                
                # ! DO NORMALIZATION HERE
                
                component.object_class = "insulator_{}".format(insulator_counter)
                insulator_counter += 1
            else:
                component.object_class = "dumper_{}".format(dumper_counter)
                dumper_counter += 1        
        
        return components

    def get_predictions_block2_concrete(self):
        pass
    
    def postprocess(self, frame, outs):
        """
        Processes data outputted from 3 YOLO layers. Removes BBs with low confidence using
        non-max suppression.
        :param frame: image section on which detection is hapenning
        :param outs: results outputted by a neural net
        :return: list of class objects representing objects detected 
        """
        # Get image width and height since object's location on an image is given as a 
        # percent values which need to be multipled by the image's shape to get actual values
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        class_ids, confidences, boxes = [], [], []
        # Check all detections from 3 YOLO layers. Discard bad ones.
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_threshold:
                    # Centre of object relatively to the upper left corner.
                    # 0-1 value (percent) multiplied by image width and height
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    # Width and height of the bounding box
                    # ! Some output issue. Fix for now. Outputs value more than 1 WTF
                    width_percent = detection[2] if detection[2] < 0.98 else 0.98
                    height_percent = detection[3] if detection[3] < 0.98 else 0.98
                    width = int(width_percent * frame_width)
                    height = int(height_percent * frame_height)
                    left = abs(int(center_x - (width / 2)))
                    top = int(center_y - (height / 2))

                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Perform non-max suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.NMS_threshold)
        
        objects_detected = list()
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # For convenience each object detected gets represented as a class object
            # ! WEIRD YOLO ISSUE. OUTPUTS INCORRECT VALUES. CAN OUTPUT MORE THAN 1 (> 100% wtf)
            right = left+width if left+width < frame_width else int(frame_width*0.99)
            bottom = top+height if top+height < frame_height else int(frame_height*0.99)
            object = DetectedObject(class_ids[i], confidences[i], left, top, right, bottom)
            objects_detected.append(object)
            
        return objects_detected
