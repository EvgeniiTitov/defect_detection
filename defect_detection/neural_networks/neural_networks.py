import cv2
import numpy as np

class Neural_Network():
    conf_threshold = 0.4
    NMS_threshold = 0.2
    # MIGHT NEED TO CHANGE TO THE HIGHER (608)
    input_width = 416
    input_height = 416
    
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
        '''
        Extracts configuration information from the txt files. 
        '''
        # ? Make this and setup NN uncallable from the outside
        with open(r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\settings.txt", "rt") as f:
            content = [line.rstrip("\n") for line in f]
        with open(r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\obj.names.txt", "rt") as f:
            classes = [line.rstrip("\n") for line in f]
        
        return (content, classes)

    def setup_network(self, configuration, weights):
        '''
        Sets up a neural network using the configuration and weights provided 
        '''
        neural_net = cv2.dnn.readNetFromDarknet(configuration, weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return neural_net

    def create_blob(self, image):
        '''
        Creates a blob that gets fed to a neural network 
        '''
        return cv2.dnn.blobFromImage(image,1/255,(self.input_width,self.input_height),
                                     [0,0,0],1,crop=False)
        
    def get_output_names(self, NN):
        '''
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106'] 
        '''
        layers_names = NN.getLayerNames()
        
        return [layers_names[i[0]-1] for i in NN.getUnconnectedOutLayers()]
        
    
    def get_predictions_block_1_images(self, image):
        '''
        works with plain images (np arrays straight away) 
        '''
        # Create a blob from the image
        blob = self.create_blob(image)
        # Pass the image to the neural network
        self.block_1_NN.setInput(blob)
        # Get output YOLO layers
        layers = self.get_output_names(self.block_1_NN)
        # Run forward pass to get output from 3 output layers (BBs). List of 3 numpy
        # matrices of shape (507, 6),(2028, 6),(8112, 6)
        output = self.block_1_NN.forward(layers)

        return self.postprocess(image, output)
        
    def get_predictions_block_1_video(self):
        '''
        processes every frame as a cap (instance of cv2.VideoCapture)
        '''
        pass
    
    def get_predictions_block_2_metal(self):
        pass

    def get_predictions_block_2_concrete(self):
        pass
    
    def postprocess(self, frame, outs):
        '''
        Processes data outputted from 3 YOLO layers. Removes BBs with low confidence using
        non-max suppression. 
        '''
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids, confidences, boxes = [],[],[]
        # Check all detections from 3 YOLO layers. Discard bad ones.
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0]*frame_width)
                    center_y = int(detection[1]*frame_height)
                    width = int(detection[2]*frame_width)
                    height = int(detection[3]*frame_height)
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
            objects_detected.append([class_ids[i], confidences[i], left, top, left+width, top+height])
            
        return objects_detected