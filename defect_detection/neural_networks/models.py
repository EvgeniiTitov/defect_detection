import cv2
import numpy as np


class NetPoles:
    """
    UTILITY POLES detecting and classifying neural network.
    predict method receives IMAGE (NumPy array)
    Returns list of the objects (utility poles) predicted.
    """

    # ! MIGHT MAKE SENSE TO MOVE THOSE PARAMETERS TO THE TXT FILE SINCE USER DOESN't WANT
    # ! TO DEAL WITH THE CODE BUT WITH A TXT FILE!

    confidence_thresh = 0.2
    NMS_thresh = 0.2
    input_width, input_height = 416, 416

    def __init__(self):
        self.config, self.weights = self.load_files()
        self.net = self.setup_net()

    def load_files(self):
        """
        Extract configuration information and weights for the net
        """
        with open(
                r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\net_poles.txt",
                "rt") as f:
            content = [line.rstrip("\n") for line in f]

        return content

    def setup_net(self):
        """
        Initializes the net with the config and weights provided.
        """
        neural_net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return neural_net

    def create_blob(self, image):
        """
        Creates a blob out of the image provided. Returns the blob
        """
        blob = cv2.dnn.blobFromImage(image, 1 / 255,
                                     (self.input_width, self.input_height),
                                     [0, 0, 0], 1, crop=False)

        return blob

    def output_layers(self, net):
        """
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106']
        """
        layers = net.getLayerNames()

        return [layers[i[0]-1] for i in net.getUnconnectedOutLayers()]

    def process_predictions(self, image, predictions):
        """
        Process all BBs predicted. Keep only the valid ones.
        """
        image_height, image_width = image.shape[0], image.shape[1]
        classIds, confidences, boxes = [], [], []
        objects_predicted = list()
        # For each prediction from each of 3 YOLO layers
        for prediction in predictions:
            # For each detection from each YOLO layer
            for detection in prediction:
                scores = detection[5:]  # Class scores
                classId = np.argmax(scores)  # Index of the class with highest probability
                confidence = scores[classId]  # Value of this BB's confidence
                if confidence > self.confidence_thresh:
                    # Centre of an object relatively to the upper left corner of the image in percent
                    centre_x = int(detection[0] * image_width)
                    centre_y = int(detection[1] * image_height)
                    # Width and height of the BB predicted in percent. Catching ERRORS:
                    width_percent = detection[2] if detection[2] < 0.98 else 0.98
                    height_percent = detection[3] if detection[3] < 0.98 else 0.98
                    # Calculate actual size of the BB
                    width = int(width_percent * image_width)
                    height = int(height_percent * image_height)
                    # ANOTHER ERROR CATCHING WITH ABS
                    left = int(centre_x - (width / 2)) if int(centre_x - (width / 2)) > 0 else 2
                    top = int(centre_y - (height / 2)) if int(centre_y - (height / 2)) > 0 else 2
                    # Save prediction results
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non-max suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.NMS_thresh)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            objects_predicted.append([classIds[i], confidences[i], left,
                                      top, left + width, top + height])

        return objects_predicted

    def predict(self, image):
        """
        Performs utility pole detection and classification.
        Accepts an image (NumPy array)
        Returns list of objects detected (List of lists) [[obj1], [obj2]]
        """
        blob = self.create_blob(image)
        # Pass the blob to the neural net
        self.net.setInput(blob)
        # Get output YOLO layers from which read predictions
        layers = self.output_layers(self.net)
        # Run forward pass and get predictions from 3 YOLO layers
        predictions = self.net.forward(layers)
        # Parse the predictions, save only the valid ones
        poles = self.process_predictions(image, predictions)

        return poles


class NetElements:
    """
    ELEMENTS detecting and classifying neural network
    Receives IMAGE (NumPy array)
    Returns list of the objects (components: INSULATORS, DUMPERS) predicted.
    """

    # ! MIGHT MAKE SENSE TO MOVE THOSE PARAMETERS TO THE TXT FILE SINCE USER DOESN't WANT
    # ! TO DEAL WITH THE CODE BUT WITH A TXT FILE!

    confidence_thresh = 0.15
    NMS_thresh = 0.25
    input_width, input_height = 608, 608

    def __init__(self):
        self.config, self.weights = self.load_files()
        self.net = self.setup_net()

    def load_files(self):
        """
        Extract configuration information and weights for the net
        """
        with open(
                r"C:\Users\Evgenii\Desktop\Python_Programming\Python_Projects\defect_detection\defect_detection\weights_configs\net_components.txt",
                "rt") as f:
            content = [line.rstrip("\n") for line in f]

        return content

    def setup_net(self):
        """
        Initializes the net with the config and weights provided.
        """
        neural_net = cv2.dnn.readNetFromDarknet(self.config, self.weights)
        neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return neural_net

    def create_blob(self, image):
        """
        Creates a blob out of the image provided. Returns the blob
        """
        blob = cv2.dnn.blobFromImage(image, 1 / 255,
                                     (self.input_width, self.input_height),
                                     [0, 0, 0], 1, crop=False)

        return blob

    def output_layers(self, net):
        """
        Returns names of the output YOLO layers: ['yolo_82', 'yolo_94', 'yolo_106']
        """
        layers = net.getLayerNames()

        return [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def process_predictions(self, image, predictions):
        """
        Process all BBs predicted. Keep only the valid ones.
        """
        image_height, image_width = image.shape[0], image.shape[1]
        classIds, confidences, boxes = [], [], []
        objects_predicted = list()
        # For each prediction from each of 3 YOLO layers
        for prediction in predictions:
            # For each detection from one YOLO layer
            for detection in prediction:
                scores = detection[5:]
                classId = np.argmax(scores)  # Index of a BB with highest confidence
                confidence = scores[classId]  # Value of this BB's confidence
                if confidence > self.confidence_thresh:
                    # Centre of object relatively to the upper left corner in percent
                    centre_x = int(detection[0] * image_width)
                    centre_y = int(detection[1] * image_height)
                    # Width and height of the BB predicted. Check for ERROR
                    width_percent = detection[2] if detection[2] < 0.98 else 0.98
                    height_percent = detection[3] if detection[3] < 0.98 else 0.98
                    # Calculate actual size of the BB
                    width = int(width_percent * image_width)
                    height = int(height_percent * image_height)
                    # ERROR CATCHING WITH ABS
                    left = abs(int(centre_x - (width / 2)))
                    top = abs(int(centre_y - (height / 2)))
                    # Save prediction results
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non-max suppression to eliminate redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.NMS_thresh)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            objects_predicted.append([classIds[i], confidences[i], left,
                                      top, left + width, top + height])

        return objects_predicted

    def predict(self, image):
        """
        Performs utility pole detection and classification. Returns list of objects detected
        """
        blob = self.create_blob(image)
        # Pass the blob to the neural net
        self.net.setInput(blob)
        # Get output YOLO layers from which read predictions
        layers = self.output_layers(self.net)
        # Run forward pass and get predictions from 3 YOLO layers
        predictions = self.net.forward(layers)
        # Parse the predictions, save only the valid ones
        components = self.process_predictions(image, predictions)

        return components

class DefectDetector_1:
    pass

class DefectDetector_2:
    pass