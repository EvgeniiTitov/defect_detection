import cv2

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
        self.block_2_NN_concrete = self.setup_network(self.configuration_3_classes, self.block_2_weights_concrete)
            
    def load_files(self):
        # ? Make this and setup NN uncallable from the outside
        with open("defect_detection\weights_configs\settings.txt", "rt") as f:
            content = [line.rstrip("\n") for line in f]
        with open("defect_detection\weights_configs\obj.names.txt", "rt") as f:
            classes = [line.rstrip("\n") for line in f]
        return (content, classes)

    def setup_network(self, configuration, weights):
        neural_net = cv2.dnn.readNetFromDarknet(configuration, weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return neural_net

    def create_blob(self, image):
        return cv2.dnn.blobFromImage(image,1/255,(self.input_width,self.input_height),
                                     [0,0,0],1,crop=False)
        
    def get_output_names(self, NN):
        '''
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106'] 
        '''
        layers_names = NN.getLayerName()
        return [layers_names[i[0]-1] for i in NN.getUnconnectedOutLayers()]
        
    
    def get_predictions_block_1_matrix(self, image):
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
        
    def get_predictions_block_1_cap(self):
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
        frame_width = frame.width[1]
        
    